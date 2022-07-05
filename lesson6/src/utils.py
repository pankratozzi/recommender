import pandas as pd
import numpy as np

cold_users = []

def prefilter_items(data, take_n_popular=5000, item_features=None):
    # Уберем самые популярные товары (их и так купят)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

    top_popular = popularity[popularity['share_unique_users'] > 0.2].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]

    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.02].item_id.tolist()
    data = data[~data['item_id'].isin(top_notpopular)]

    # Уберем не интересные для рекоммендаций категории (department)
    if item_features is not None:
        department_size = pd.DataFrame(
            item_features.groupby('department')['item_id'].nunique().sort_values(ascending=False)).reset_index()
        department_size.columns = ['department', 'n_items']
        rare_departments = department_size[department_size['n_items'] < 150].department.tolist()
        items_in_rare_departments = item_features[
            item_features['department'].isin(rare_departments)].item_id.unique().tolist()

        data = data[~data['item_id'].isin(items_in_rare_departments)]

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    data = data[data['price'] > 2]

    # Уберем слишком дорогие товарыs
    data = data[data['price'] < 50]

    # уберем товары, не продававшиеся более 12-18 месяцев
    data = data[data['week_no'] >= data['week_no'].max() - 52]

    # Возьмем топ по популярности
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()

    # Заведем фиктивный item_id (если юзер покупал товары из топ-N, то он "купил" такой товар)
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999

    return data


def reduce_memory(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and str(col_type)[:4] != 'uint' and str(col_type) != 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif str(col_type)[:4] != 'uint':
            df[col] = df[col].astype('category')
    return df


def popularity_recommendation(data, n=5):
    """Топ-n популярных товаров"""

    popular = data.groupby('item_id')['sales_value'].sum().reset_index()
    popular.sort_values('sales_value', ascending=False, inplace=True)

    recs = popular.head(n).item_id

    return recs.tolist()


def postfilter(recommendations, item_info, N=5):
    """Пост-фильтрация товаров

    Input
    -----
    recommendations: list
        Ранжированный список item_id для рекомендаций
    item_info: pd.DataFrame
        Датафрейм с информацией о товарах
    """

    # Уникальность
    unique_recommendations = []
    [unique_recommendations.append(item) for item in recommendations if item not in unique_recommendations]

    # Разные категории
    categories_used = []
    final_recommendations = []
    CATEGORY_NAME = 'sub_commodity_desc'
    for item in unique_recommendations:
        category = item_features.loc[item_features['item_id'] == item, CATEGORY_NAME].values[0]

        if category not in categories_used:
            final_recommendations.append(item)

        unique_recommendations.remove(item)
        categories_used.append(category)

    n_rec = len(final_recommendations)
    if n_rec < N:
        final_recommendations.extend(unique_recommendations[:N - n_rec])
    else:
        final_recommendations = final_recommendations[:N]

    assert len(final_recommendations) == N, 'Количество рекомендаций != {}'.format(N)
    return final_recommendations

def rule(x, y, model, N=5):
    if x in y:
        return recommender.overall_top_purchases[:N]
    if model == 'als':
        return recommender.get_als_recommendations(x, N=N)
    elif model == 'own':
        return recommender.get_own_recommendations(x, N=N)
    elif model == 'similar_items':
        return recommender.get_similar_items_recommendation(x, N=N)
    elif model == 'similar_users':
        return recommender.get_similar_users_recommendation(x, N=N)
    elif model == 'bayesian':
        return recommender.get_bayesian_recommendations(x, N=N)

def rerank(user_id, N, out=cold_users):
    if user_id in df_predict.user_id:
        return df_predict[df_predict['user_id']==user_id].sort_values('proba_item_purchase', ascending=False).head(N).item_id.tolist()
    else:
        return rule(user_id, cold_users, model='own', N=5)

def rerank_post(user_id, N_rank=20, N_post=5):
    try:
        out = rerank(user_id, N=N_rank)
        out = postfilter(out, item_features, N=N_post)
    except AssertionError:
        out = rule(user_id, cold_users, model='own', N=N_post)
    return out

def transform_data_for_eval(dataset, rec_col, user_col='user_id'):
    '''
    Func for transforming recommendations into kaggle evaluation format

    Parameters:
    dataset (pd.DataFrame): Dataset with 2 required columns:
        rec_col - column with recommendations should be iterable
        user_col - columns with user id

    rec_col (str): name of column in dataset with recommendations

    user_col (str): name of column in dataset with user id

    Returns:
    pd.DataFrame: DataFrame in suitable format

   '''
    eval_dataset = dataset[[user_col, rec_col]].copy()
    eval_dataset[rec_col] = eval_dataset[rec_col].apply(lambda x: ' '.join([str(i) for i in x]))
    eval_dataset.rename(columns={
        user_col: 'UserId',
        rec_col: 'Predicted'
    }, inplace=True)
    return eval_dataset
