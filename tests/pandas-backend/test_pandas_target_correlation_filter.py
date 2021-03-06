import unittest

import pandas as pd

from featurefilter import TargetCorrelationFilter


def test_low_continuous_correlation():
    train_df = pd.DataFrame({'A': [0, 0, 1, 1], 'Y': [0, 1, 0, 1]})

    target_correlation_filter = TargetCorrelationFilter(target_column='Y')
    train_df = target_correlation_filter.fit(train_df)

    assert target_correlation_filter.columns_to_drop == ['A']


def test_high_negative_continuous_correlation():
    train_df = pd.DataFrame({'A': [0, 0], 'B': [1, 0], 'Y': [0, 1]})
    test_df = pd.DataFrame({'A': [0, 1], 'B': [1, 1], 'Y': [0, 1]})

    target_correlation_filter = TargetCorrelationFilter(target_column='Y')
    train_df = target_correlation_filter.fit_transform(train_df)
    test_df = target_correlation_filter.transform(test_df)

    # Make sure column 'B' is dropped for both train and test set
    # Also, column 'A' must not be dropped for the test set even though its
    # correlation in the test set is above the threshold
    assert train_df.equals(pd.DataFrame({'A': [0, 0], 'Y': [0, 1]}))
    assert test_df.equals(pd.DataFrame({'A': [0, 1], 'Y': [0, 1]}))


def test_high_positive_continuous_correlation():
    train_df = pd.DataFrame({'A': [0, 0], 'B': [0, 1], 'Y': [0, 1]})
    test_df = pd.DataFrame({'A': [0, 1], 'B': [1, 1], 'Y': [0, 1]})

    target_correlation_filter = TargetCorrelationFilter(target_column='Y')
    train_df = target_correlation_filter.fit_transform(train_df)
    test_df = target_correlation_filter.transform(test_df)

    # Make sure column 'B' is dropped for both train and test set
    # Also, column 'A' must not be dropped for the test set even though its
    # correlation in the test set is above the threshold
    assert train_df.equals(pd.DataFrame({'A': [0, 0], 'Y': [0, 1]}))
    assert test_df.equals(pd.DataFrame({'A': [0, 1], 'Y': [0, 1]}))


def test_low_categorical_correlation():
    train_df = pd.DataFrame({'A': ['a', 'a'], 'B': ['b', 'a'], 'Y': [0, 1]})
    test_df = pd.DataFrame({'A': ['a', 'b'], 'B': ['b', 'b'], 'Y': [0, 1]})

    target_correlation_filter = TargetCorrelationFilter(target_column='Y')
    train_df = target_correlation_filter.fit_transform(train_df)
    test_df = target_correlation_filter.transform(test_df)

    # Make sure column 'B' is dropped for both train and test set
    # Also, column 'A' must not be dropped for the test set even though its
    # correlation in the test set is above the threshold
    assert train_df.equals(pd.DataFrame({'A': ['a', 'a'], 'Y': [0, 1]}))
    assert test_df.equals(pd.DataFrame({'A': ['a', 'b'], 'Y': [0, 1]}))


def test_sample_ratio():
    train_df = pd.DataFrame({'A': [0, 0, 1, 1, 1, 1], 'Y': [0, 1, 0, 1, 0, 1]})

    # Set seed to sample the following DataFrame for a correlation of ~0.33:
    #    A  Y
    # 5  1  1
    # 2  1  0
    # 1  0  1
    # 3  1  1
    target_correlation_filter_1 = TargetCorrelationFilter(target_column='Y',
                                                          sample_ratio=0.7,
                                                          seed=0)
    target_correlation_filter_1.fit(train_df)
    # Set seed to sample the following DataFrame for a correlation of 0.00:
    #    A  Y
    # 0  0  0
    # 2  1  0
    # 5  1  1
    # 1  0  1
    target_correlation_filter_2 = TargetCorrelationFilter(target_column='Y',
                                                          sample_ratio=0.7,
                                                          seed=8)
    target_correlation_filter_2.fit(train_df)
    # Set seed to sample the following DataFrame for a correlation of ~0.99:
    #    A  Y
    # 2  1  0
    # 1  0  1
    target_correlation_filter_3 = TargetCorrelationFilter(target_column='Y',
                                                          sample_ratio=0.4,
                                                          seed=1)
    target_correlation_filter_3.fit(train_df)

    assert target_correlation_filter_1.columns_to_drop == []
    assert target_correlation_filter_2.columns_to_drop == ['A']
    assert target_correlation_filter_3.columns_to_drop == ['A']


def test_sparse_data():
    # Create a sparse DataFrame with a correlation of -1
    sdf = pd.DataFrame({'A': [0] * 10 + [1] * 10,
                        'Y': [1] * 10 + [0] * 10}).to_sparse()

    target_correlation_filter = TargetCorrelationFilter(target_column='Y')
    target_correlation_filter.fit(sdf)

    assert target_correlation_filter.columns_to_drop == ['A']


if __name__ == '__main__':
    unittest.main()
