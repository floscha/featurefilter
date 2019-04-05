import unittest

import pandas as pd

from featurefilter import FeatureCorrelationFilter


def test_fit_high_continuous_correlation():
    train_df = pd.DataFrame({'A': [0, 1],
                             'B': [0, 1]})

    filter = FeatureCorrelationFilter()
    train_df = filter.fit(train_df)

    assert filter.columns_to_drop == ['B']


def test_excluding_target_column():
    train_df = pd.DataFrame({'A': [0, 1],
                             'B': [0, 1],
                             'Y': [0, 1]})

    filter = FeatureCorrelationFilter(target_column='Y')
    train_df = filter.fit(train_df)

    assert filter.columns_to_drop == ['B']


def test_high_negative_continuous_correlation():
    train_df = pd.DataFrame({'A': [0, 1], 'B': [0, -1], 'Y': [0, 1]})
    test_df = pd.DataFrame({'A': [0, 0], 'B': [0, 0], 'Y': [0, 1]})

    filter = FeatureCorrelationFilter(target_column='Y')
    train_df = filter.fit_transform(train_df)
    test_df = filter.transform(test_df)

    assert train_df.equals(pd.DataFrame({'A': [0, 1], 'Y': [0, 1]}))
    assert test_df.equals(pd.DataFrame({'A': [0, 0], 'Y': [0, 1]}))


def test_categorical_correlation():
    train_df = pd.DataFrame({'A': ['a', 'b'], 'B': ['a', 'b'], 'Y': [0, 1]})
    test_df = pd.DataFrame({'A': ['a', 'b'], 'B': ['b', 'b'], 'Y': [0, 1]})

    filter = FeatureCorrelationFilter(target_column='Y')
    train_df = filter.fit_transform(train_df)
    test_df = filter.transform(test_df)

    assert train_df.equals(pd.DataFrame({'A': ['a', 'b'], 'Y': [0, 1]}))
    assert test_df.equals(pd.DataFrame({'A': ['a', 'b'], 'Y': [0, 1]}))


if __name__ == '__main__':
    unittest.main()
