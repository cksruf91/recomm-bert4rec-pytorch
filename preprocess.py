import argparse
import os
import random
from itertools import accumulate
from typing import Tuple, Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from common.loading_functions import loading_data
from common.utils import DefaultDict, progressbar
from config import CONFIG

random.seed(42)


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='1M', choices=['10M', '1M', 'BRUNCH'], help='데이터셋', type=str)
    return parser.parse_args()


def uniform_random_sample(n, exclude_items, items):
    sample = []
    while len(sample) < n:
        n_item = random.choice(items)
        if n_item in exclude_items:
            continue
        if n_item in sample:
            continue
        sample.append(n_item)
    assert len(sample) == n
    return sample


def weighted_random_sample(n, exclude_items, items, cum_sums):
    n_items = len(exclude_items)
    samples = random.choices(
        items, cum_weights=cum_sums, k=n_items + n + 100
    )

    sample = list(set(samples) - exclude_items)
    sample = sample[:n]
    assert len(sample) == n
    return sample


def get_negative_samples(train_df, test_df, n_sample=99, method='random'):
    negative_sampled_test = []

    # 샘플링을 위한 아이템들의 누적합
    # train_df.loc[:,'item_count'] = 1
    train_df = train_df.assign(item_count=1)
    item_counts = train_df.groupby('item_id')['item_count'].sum().reset_index()
    item_counts['cumulate_count'] = [c for c in accumulate(item_counts.item_count)]

    # 샘플링을 위한 변수
    item_list = item_counts['item_id'].tolist()
    item_cumulate_count = item_counts['cumulate_count'].tolist()

    # 유저가 이전에 interaction 했던 아이템들
    user_interactions = train_df.groupby('user_id')['item_id'].agg(lambda x: set(x.tolist()))

    for uid, iid in zip(test_df['user_id'].tolist(), test_df['item_id'].tolist()):
        row = [iid]

        try:
            inter_items = user_interactions[uid]
        except KeyError as e:
            inter_items = set([])

        if method == 'random':
            sample = uniform_random_sample(n_sample, inter_items, item_list)
        elif method == 'weighted':
            sample = weighted_random_sample(n_sample, inter_items, item_list, cum_sums=item_cumulate_count)
        else:
            raise ValueError(f"invalid sampling method {method}")

        row.extend(sample)

        negative_sampled_test.append(row)

    return negative_sampled_test


def last_session_test_split(df: DataFrame, user_col: str, time_col: str) -> Tuple[DataFrame, DataFrame]:
    """ 학습 테스트 데이터 분리 함수
    각 유저별 마지막 interaction 읕 테스트로 나머지를 학습 데이터셋으로 사용

    Args:
        df: 전체 데이터
        user_col: 기준 유저 아이디 컬럼명
        time_col: 기준 아이템 아이디 컬럼명

    Returns: 학습 데이터셋, 테스트 데이터셋
    """

    last_action_time = df.groupby(user_col)[time_col].transform('max')

    test = df[df[time_col] == last_action_time]
    train = df[df[time_col] != last_action_time]

    test = test.groupby(user_col).first().reset_index()

    print(f'test set size : {len(test)}')
    user_list = train[user_col].unique()
    drop_index = test[test[user_col].isin(user_list) == False].index
    test.drop(drop_index, inplace=True)
    print(f'-> test set size : {len(test)}')

    return train, test


def movielens_preprocess(interactions, items, users):
    interactions.sort_values(['UserID', 'Timestamp'], inplace=True)

    item_count = interactions.groupby('MovieID')['Rating'].transform('count')
    interactions = interactions[item_count > 3].copy()

    # MovieID -> item_id
    movie_ids = set(interactions['MovieID'].unique().tolist())
    movie_id_mapper = DefaultDict(None, {
        movie_id: item_id + 1 for item_id, movie_id in enumerate(movie_ids)
    })

    interactions['item_id'] = interactions['MovieID'].map(lambda x: movie_id_mapper[x])
    items['item_id'] = items['MovieID'].map(lambda x: movie_id_mapper[x])

    # UserID -> user_id
    user_ids = set(interactions['UserID'].unique().tolist())
    user_id_mapper = DefaultDict(None, {
        user_id: user_index_id for user_index_id, user_id in enumerate(user_ids)
    })
    
    interactions['user_id'] = interactions['UserID'].map(lambda x: user_id_mapper[x])
    users['user_id'] = users['UserID'].map(lambda x: user_id_mapper[x])

    interactions = interactions[['user_id', 'item_id', 'Timestamp', 'Rating']]

    # train test split
    train, test = last_session_test_split(interactions, user_col='user_id', time_col='Timestamp')
    test['negative_sample'] = get_negative_samples(train, test)

    # create PAD, MASK id 
    token_ids = pd.DataFrame.from_dict(
        {'MovieID': [0, int(items['MovieID'].max() + 1)], 
         'Title':['[PAD]', '[MASK]'], 
         'Genres':['[PAD]', '[MASK]'], 
         'item_id':[0, int(items['item_id'].max() + 1)]}
    )
    items = pd.concat(
        [items, token_ids]
    )
    
    return train, test, items, users


def preprocess_data(data_type: str, interactions: DataFrame, items: DataFrame, users: DataFrame):
    if data_type == '1M':
        loading_function = movielens_preprocess
    else:
        raise ValueError(f"unknown data type {data_type}")

    return loading_function(interactions, items, users)


if __name__ == '__main__':
    argument = args()

    log_data, item_meta, user_meta = loading_data(argument.dataset)
    train, test, item_meta, user_meta = preprocess_data(
        argument.dataset, log_data, item_meta, user_meta
    )

    # test data negative sampling
    print(f'train data size : {len(train)}, test data size : {len(test)}')
    print(f'total item size : {len(item_meta)}, total user size : {len(user_meta)}')

    save_dir = os.path.join(CONFIG.DATA, argument.dataset)
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    train.to_csv(os.path.join(save_dir, 'train.tsv'), sep='\t', index=False)
    test.to_csv(os.path.join(save_dir, 'test.tsv'), sep='\t', index=False)
    item_meta.to_csv(os.path.join(save_dir, 'item_meta.tsv'), sep='\t', index=False)
    user_meta.to_csv(os.path.join(save_dir, 'user_meta.tsv'), sep='\t', index=False)
