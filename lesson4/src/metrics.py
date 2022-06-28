import numpy as np


def precision_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]

    flags = np.isin(bought_list, recommended_list)
    precision = flags.sum() / len(recommended_list)

    return precision


def ap_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    flags = np.isin(recommended_list, bought_list)

    if sum(flags) == 0:
        return 0

    sum_ = 0
    for i in range(k):

        if flags[i]:
            p_k = precision_at_k(recommended_list, bought_list, k=i + 1)
            sum_ += p_k

    result = sum_ / sum(flags)

    return result


def recall_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    flags = np.isin(bought_list, recommended_list)

    recall = flags.sum() / len(bought_list)

    return recall


def map_k(recommend_list, bought_list, k=5):
    return np.mean([ap_k(rec, bt, k) for rec, bt in zip(recommend_list, bought_list)])