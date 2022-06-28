import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
from implicit.nearest_neighbours import ItemItemRecommender
from implicit.bpr import BayesianPersonalizedRanking


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data, data_type='quantity', weighting=True, normalize=False, alpha=1.):
        self.data_type = data_type
        self.normalize = normalize
        self.alpha = alpha

        # Топ покупок каждого юзера
        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != 999999]

        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != 999999]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()

        self.user_item_matrix = self._prepare_matrix(data, data_type=self.data_type, normalize=self.normalize)
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self._prepare_dicts(
            self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix, K1=150, B=0.8)  # default: 100, 0.8 ver. 0.5.2
            # self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.model = self.fit(self.user_item_matrix, alpha=self.alpha)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
        self.ranker = self.fit_ranker(self.user_item_matrix)

    @staticmethod
    def _prepare_matrix(data, data_type, normalize=False):
        """Готовит user-item матрицу"""
        if data_type == 'quantity':
            user_item_matrix = pd.pivot_table(data,
                                              index='user_id', columns='item_id',
                                              values='quantity',
                                              aggfunc='count',
                                              fill_value=0
                                              )
        elif data_type == 'sales':
            user_item_matrix = pd.pivot_table(data,
                                              index='user_id', columns='item_id',
                                              values='sales_value',
                                              aggfunc='sum',
                                              fill_value=0
                                              )
            if normalize:
                user_item_matrix = user_item_matrix / user_item_matrix.max()  # normalize
        elif data_type == 'quantity_sum':
            user_item_matrix = pd.pivot_table(data,
                                              index='user_id', columns='item_id',
                                              values='quantity',
                                              aggfunc='sum',
                                              fill_value=0
                                              )
            if normalize:
                user_item_matrix = user_item_matrix / user_item_matrix.max()
        else:
            raise ValueError(f'Agg data type must be "quantity", "sales" or "quantity_sum", given: {data_type}')

        user_item_matrix = user_item_matrix.astype(float)

        return user_item_matrix

    @staticmethod
    def _prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).tocsr())  # ver. 0.5.2
        # own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=50, regularization=0.001, iterations=15, num_threads=4, alpha=1.):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads,
                                        use_gpu=False,
                                        random_state=42)
        model.fit(csr_matrix(user_item_matrix).tocsr() * alpha)  # ver. 0.5.2
        # model.fit(csr_matrix(user_item_matrix).T.tocsr()*alpha)

        return model

    @staticmethod
    def fit_ranker(user_item_matrix, factors=50, learning_rate=0.03, regularization=0.01, iterations=200):
        ranker = BayesianPersonalizedRanking(factors=factors,
                                             learning_rate=learning_rate,
                                             regularization=regularization,
                                             iterations=iterations,
                                             num_threads=4,
                                             random_state=42)
        ranker.fit(csr_matrix(user_item_matrix).tocsr())

        return ranker

    def _update_dict(self, user_id):
        """Если появился новыю user / item, то нужно обновить словари"""

        if user_id not in self.userid_to_id.keys():
            max_id = max(list(self.userid_to_id.values()))
            max_id += 1

            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})

    def _get_similar_item(self, item_id):
        """Находит товар, похожий на item_id"""
        recs = self.model.similar_items(self.itemid_to_id[item_id], N=2)  # Товар похож на себя -> рекомендуем 2 товара
        top_rec = recs[0][1]  # ver. 0.5.2
        # top_rec = recs[1][0]
        return self.id_to_itemid[top_rec]

    def _extend_with_top_popular(self, recommendations, N=5):
        """Если кол-во рекоммендаций < N, то дополняем их топ-популярными"""

        if len(recommendations) < N:
            recommendations.extend(self.overall_top_purchases[:N])
            recommendations = recommendations[:N]

        return recommendations

    def _get_recommendations(self, user, model, N=5):
        """Рекомендации через стардартные библиотеки implicit"""

        self._update_dict(user_id=user)
        user_id = self.userid_to_id[user]

        """ ver. 0.5.2 """

        model_name = model.__class__.__name__

        if model_name == 'ItemItemRecommender':

            res = model.recommend(userid=user_id,
                                  user_items=csr_matrix(self.user_item_matrix).tocsr()[user_id, :],
                                  N=N - 1,
                                  filter_already_liked_items=False,
                                  filter_items=[self.itemid_to_id[999999]],
                                  recalculate_user=True)[0].tolist()
            res = [self.id_to_itemid[rec] for rec in res]

        elif model_name == 'AlternatingLeastSquares':
            res = [self.id_to_itemid[rec] for rec in model.recommend(userid=user_id,
                                                                     user_items=csr_matrix(
                                                                         self.user_item_matrix).tocsr()[user_id, :],
                                                                     N=N,
                                                                     filter_already_liked_items=False,
                                                                     filter_items=[self.itemid_to_id[999999]],
                                                                     recalculate_user=True)[0]]

        elif model_name == 'BayesianPersonalizedRanking':
            res = [self.id_to_itemid[rec] for rec in model.recommend(userid=user_id,
                                                                     user_items=csr_matrix(
                                                                         self.user_item_matrix).tocsr()[user_id, :],
                                                                     N=N,
                                                                     filter_already_liked_items=False,
                                                                     filter_items=[self.itemid_to_id[999999]],
                                                                     )[0]]
        """
        res = [self.id_to_itemid[rec[0]] for rec in model.recommend(userid=user_id,
                                                                    user_items=csr_matrix(self.user_item_matrix).tocsr(),
                                                                    N=N,
                                                                    filter_already_liked_items=False,
                                                                    filter_items=[self.itemid_to_id[999999]],
                                                                    recalculate_user=True)]
        """
        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_als_recommendations(self, user, N=5):
        """Рекомендации через стардартные библиотеки implicit"""

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.model, N=N)

    def get_bayesian_recommendations(self, user, N=5):
        """ Рекомендации на основе модели ранжирования """

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.ranker, N=N)

    def get_own_recommendations(self, user, N=5):
        """Рекомендуем товары среди тех, которые юзер уже купил"""

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.own_recommender, N=N)

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        top_users_purchases = self.top_purchases[self.top_purchases['user_id'] == user].head(N)

        res = top_users_purchases['item_id'].apply(lambda x: self._get_similar_item(x)).tolist()
        if 999999 in res: res.remove(999999)  # prev. ver.

        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами: берем N похожих пользователей и с помощью трюка рекомендуем юзеру их топ товары"""
        res = []

        # Находим топ-N похожих пользователей
        similar_users = self.model.similar_users(self.userid_to_id[user], N=N + 1)
        similar_users = [rec for rec in similar_users[0]]  # ver. 0.5.2
        # similar_users = [rec[0] for rec in similar_users]
        similar_users = similar_users[1:]  # удалим юзера из запроса

        for user in similar_users:
            user = self.id_to_userid[user]  ## нужно подать для предикта оригинальный идентификатор # ver. 0.4.8
            res.extend(self.get_own_recommendations(user, N=1))
        res = pd.Series(res).drop_duplicates().tolist()

        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res


if __name__ == '__main__':
    print(implicit.__version__)
