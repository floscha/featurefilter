import unittest

import databricks.koalas as ks
import numpy as np

from featurefilter import NaFilter


def test_fit_returns_none():
    train_df = ks.DataFrame({'A': [0, np.nan, np.nan],
                             'B': [0, 0, np.nan]})

    na_filter = NaFilter()
    return_value = na_filter.fit(train_df)

    assert return_value is None


def test_fit_sets_correct_columns_to_drop():
    train_df = ks.DataFrame({'A': [0, np.nan, np.nan],
                             'B': [0, 0, np.nan]})

    na_filter = NaFilter(max_na_ratio=0.5)
    na_filter.fit(train_df)

    assert na_filter.columns_to_drop == ['A']


def test_transform():
    test_df = ks.DataFrame({'A': [0, np.nan, np.nan],
                            'B': [0, 0, np.nan]})

    na_filter = NaFilter(max_na_ratio=0.5)
    na_filter.columns_to_drop = ['A']
    test_df = na_filter.transform(test_df)

    assert repr(test_df) == repr(ks.DataFrame({'B': [0, 0, np.nan]}))


def test_sample_ratio():
    train_df = ks.DataFrame({'A': [0, np.nan, np.nan, 0, np.nan, np.nan]})

    # Use seed to get values [0, NaN, 0, NaN] resulting in an na_ratio of 0.5
    na_filter_1 = NaFilter(max_na_ratio=0.5, sample_ratio=0.5, seed=1)
    na_filter_1.fit(train_df)
    # Use seed to get values [0, NaN, NaN] resulting in an na_ratio of 0.667
    na_filter_2 = NaFilter(max_na_ratio=0.5, sample_ratio=0.5, seed=2)
    na_filter_2.fit(train_df)

    assert na_filter_1.columns_to_drop == []
    assert na_filter_2.columns_to_drop == ['A']


# TODO Add unit test for sparse data as soon as Koalas adds sparsity support
# def test_sparse_data():
#     sdf = ks.DataFrame({'A': [0] * 10 + [np.nan] * 20,
#                         'B': [0] * 20 + [np.nan] * 10}).to_sparse()
#
#     na_filter = NaFilter(max_na_ratio=0.5)
#     na_filter.fit(sdf)
#
#     assert na_filter.columns_to_drop == ['A']


if __name__ == '__main__':
    unittest.main()
