import unittest

import pandas as pd

from featurefilter import FeatureCorrelationFilter


def test_fit_high_continuous_correlation():
    train_df = pd.DataFrame({'A': [0, 1],
                             'B': [0, 1]})

    filter_ = FeatureCorrelationFilter()
    train_df = filter_.fit(train_df)

    assert filter_.columns_to_drop == ['B']


def test_excluding_target_column():
    train_df = pd.DataFrame({'A': [0, 1],
                             'B': [0, 1],
                             'Y': [0, 1]})

    filter_ = FeatureCorrelationFilter(target_column='Y')
    train_df = filter_.fit(train_df)

    assert filter_.columns_to_drop == ['B']


def test_high_negative_continuous_correlation():
    train_df = pd.DataFrame({'A': [0, 1], 'B': [0, -1], 'Y': [0, 1]})
    test_df = pd.DataFrame({'A': [0, 0], 'B': [0, 0], 'Y': [0, 1]})

    filter_ = FeatureCorrelationFilter(target_column='Y')
    train_df = filter_.fit_transform(train_df)
    test_df = filter_.transform(test_df)

    assert train_df.equals(pd.DataFrame({'A': [0, 1], 'Y': [0, 1]}))
    assert test_df.equals(pd.DataFrame({'A': [0, 0], 'Y': [0, 1]}))


def test_categorical_correlation():
    train_df = pd.DataFrame({'A': ['a', 'b'], 'B': ['a', 'b'], 'Y': [0, 1]})
    test_df = pd.DataFrame({'A': ['a', 'b'], 'B': ['b', 'b'], 'Y': [0, 1]})

    filter_ = FeatureCorrelationFilter(target_column='Y')
    train_df = filter_.fit_transform(train_df)
    test_df = filter_.transform(test_df)

    assert train_df.equals(pd.DataFrame({'A': ['a', 'b'], 'Y': [0, 1]}))
    assert test_df.equals(pd.DataFrame({'A': ['a', 'b'], 'Y': [0, 1]}))


def test_sample_ratio():
    train_df = pd.DataFrame({'A': [0, 0, 1, 1, 1, 1], 'B': [0, 1, 0, 1, 0, 1]})

    # Set seed to sample the following DataFrame for a correlation of 0.00:
    #    A  Y
    # 0  0  0
    # 2  1  0
    # 5  1  1
    # 1  0  1
    filter_1 = FeatureCorrelationFilter(sample_ratio=0.7, seed=8)
    filter_1.fit(train_df)

    # Set seed to sample the following DataFrame for a correlation of ~0.99:
    #    A  Y
    # 2  1  0
    # 1  0  1
    filter_2 = FeatureCorrelationFilter(sample_ratio=0.4, seed=1)
    filter_2.fit(train_df)

    assert filter_1.columns_to_drop == []
    assert filter_2.columns_to_drop == ['B']


if __name__ == '__main__':
    unittest.main()
