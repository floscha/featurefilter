import unittest

import pandas as pd

from featurefilter import TreeBasedFilter


def test_fit_returns_none():
    train_df = pd.DataFrame({'A': [0, 0, 1, 1],
                             'B': [0, 1, 0, 1],
                             'Y': [0, 0, 1, 1]})

    tree_based_filter = TreeBasedFilter(target_column='Y')
    return_value = tree_based_filter.fit(train_df)

    assert return_value is None

def test_fit_sets_correct_columns_to_drop_with_continuous_target():
    train_df = pd.DataFrame({'A': [0, 0, 1, 1],
                             'B': [0, 1, 0, 1],
                             'Y': [0, 0, 1, 1]})

    tree_based_filter = TreeBasedFilter(target_column='Y',
                                        top_features=1)
    tree_based_filter.fit(train_df)

    assert tree_based_filter.columns_to_drop == ['B']

def test_fit_sets_correct_columns_to_drop_with_categorical_target():
    train_df = pd.DataFrame({'A': [0, 0, 1, 1],
                             'B': [0, 1, 0, 1],
                             'Y': ['a', 'a', 'b', 'b']})

    tree_based_filter = TreeBasedFilter(target_column='Y',
                                        categorical_target=True,
                                        top_features=1)
    tree_based_filter.fit(train_df)

    assert tree_based_filter.columns_to_drop == ['B']

def test_setting_model_parameters():
    model_parameters = {'max_depth': 10}
    train_df = pd.DataFrame({'A': [0, 0, 1, 1],
                             'B': [0, 1, 0, 1],
                             'Y': [0, 0, 1, 1]})

    tree_based_filter = TreeBasedFilter(target_column='Y',
                                        model_parameters=model_parameters)
    tree_based_filter.fit(train_df)

    assert tree_based_filter._model.max_depth == 10

def test_fit_transform():
    train_df = pd.DataFrame({'A': [0, 0, 1, 1],
                             'B': [0, 1, 0, 1],
                             'Y': [0, 0, 1, 1]})
    expected_output = pd.DataFrame({'A': [0, 0, 1, 1],
                                    'Y': [0, 0, 1, 1]})

    tree_based_filter = TreeBasedFilter(target_column='Y',
                                        top_features=1)
    train_df = tree_based_filter.fit_transform(train_df)

    assert train_df.equals(expected_output)


if __name__ == '__main__':
    unittest.main()
