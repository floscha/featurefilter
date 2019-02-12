import unittest

import numpy as np
import pandas as pd

from featurefilter import VarianceFilter


def test_fit_returns_none():
    train_df = pd.DataFrame({'A': [0., 1.], 'B': [0., 0.]})

    variance_filter = VarianceFilter()
    return_value = variance_filter.fit(train_df)

    assert return_value is None


def test_fit_sets_correct_columns_to_drop():
    train_df = pd.DataFrame({'A': [0., 1.], 'B': [0., 0.]})

    variance_filter = VarianceFilter()
    variance_filter.fit(train_df)

    assert variance_filter.columns_to_drop == ['B']


def test_transform():
    test_df = pd.DataFrame({'A': [0., 0.], 'B': [0., 1.]})

    variance_filter = VarianceFilter()
    variance_filter.columns_to_drop = ['B']
    test_df = variance_filter.transform(test_df)

    assert test_df.equals(pd.DataFrame({'A': [0., 0.]}))


def test_fit_transform_continuous():
    train_df = pd.DataFrame({'A': [0., 1.], 'B': [0., 0.]})

    variance_filter = VarianceFilter()
    train_df = variance_filter.fit_transform(train_df)

    assert train_df.equals(pd.DataFrame({'A': [0., 1.]}))


def test_sample_ratio():
    train_df = pd.DataFrame({'A': [0, 0, 1]})

    # Set seed to consider a sample of [0, 1]
    variance_filter_1 = VarianceFilter(sample_ratio=0.7, seed=1)
    variance_filter_1.fit(train_df)
    # Set seed to consider a sample of [1, 1]
    variance_filter_2 = VarianceFilter(sample_ratio=0.7, seed=3)
    variance_filter_2.fit(train_df)

    assert variance_filter_1.columns_to_drop == []
    assert variance_filter_2.columns_to_drop == ['A']


def test_remove_min_variance_for_categorical():
    train_df = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['a', 'a', 'a']})
    test_df = pd.DataFrame({'A': ['a', 'a', 'b'], 'B': ['a', 'b', 'c']})

    variance_filter = VarianceFilter(unique_cut=50)
    train_df = variance_filter.fit_transform(train_df)
    test_df = variance_filter.transform(test_df)

    # Make sure column 'B' is dropped for both train and test set
    # Also, column 'A' must not be dropped for the test set even though its
    # variance in the test set is below the threshold
    assert train_df.equals(pd.DataFrame({'A': ['a', 'b', 'c']}))
    assert test_df.equals(pd.DataFrame({'A': ['a', 'a', 'b']}))


def test_remove_min_variance_for_single_valued_variables():
    "Make sure it does not crash for variables with only one value"
    train_df = pd.DataFrame({'A': ['a'] * 100})

    variance_filter = VarianceFilter()
    train_df = variance_filter.fit_transform(train_df)

    # Make sure column 'B' is dropped for both train and test set
    # Also, column 'A' must not be dropped for the test set even though its
    # variance in the test set is below the threshold
    assert np.array_equal(train_df.values, np.empty((100, 0)))


if __name__ == '__main__':
    unittest.main()
