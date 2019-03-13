import contextlib
from io import StringIO
import unittest

import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestClassifier,
                              RandomForestRegressor)

from featurefilter import TreeBasedFilter


def test_decision_tree_is_default_model():
    tree_based_filter = TreeBasedFilter(target_column='')

    model = tree_based_filter._model
    assert isinstance(model, DecisionTreeRegressor)


def test_invalid_model_type():
    invalid_model_type = 'UNK'
    with pytest.raises(ValueError) as excinfo:
        TreeBasedFilter(target_column='', model_type=invalid_model_type)

    assert str(excinfo.value).startswith("Model '%s' not available."
                                         % invalid_model_type)


def test_gradient_boosting_regressor_model_type():
    tree_based_filter = TreeBasedFilter(target_column='',
                                        model_type='GradientBoosting')

    model = tree_based_filter._model
    assert isinstance(model, GradientBoostingRegressor)


def test_gradient_boosting_classifier_model_type():
    tree_based_filter = TreeBasedFilter(target_column='',
                                        model_type='GradientBoosting',
                                        categorical_target=True)

    model = tree_based_filter._model
    assert isinstance(model, GradientBoostingClassifier)


def test_random_forest_regressor_model_type():
    tree_based_filter = TreeBasedFilter(target_column='',
                                        model_type='RandomForest')

    model = tree_based_filter._model
    assert isinstance(model, RandomForestRegressor)


def test_random_forest_classifier_model_type():
    tree_based_filter = TreeBasedFilter(target_column='',
                                        model_type='RandomForest',
                                        categorical_target=True)

    model = tree_based_filter._model
    assert isinstance(model, RandomForestClassifier)


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


def test_fit_transform_top_features():
    train_df = pd.DataFrame({'A': [0, 0, 1, 1],
                             'B': [0, 1, 0, 1],
                             'Y': [0, 0, 1, 1]})
    expected_output = pd.DataFrame({'A': [0, 0, 1, 1],
                                    'Y': [0, 0, 1, 1]})

    tree_based_filter = TreeBasedFilter(target_column='Y', top_features=1)
    train_df = tree_based_filter.fit_transform(train_df)

    assert train_df.equals(expected_output)


def test_fit_transform_threshold():
    train_df = pd.DataFrame({'A': [0, 0, 1, 1],
                             'B': [0, 1, 0, 1],
                             'Y': [0, 0, 1, 1]})
    expected_output = pd.DataFrame({'A': [0, 0, 1, 1],
                                    'Y': [0, 0, 1, 1]})

    tree_based_filter = TreeBasedFilter(target_column='Y', threshold=0.5)
    train_df = tree_based_filter.fit_transform(train_df)

    assert train_df.equals(expected_output)


def _capture_verbose_output_for_model(filter):
    df = pd.DataFrame({'A': [0, 0, 1, 1],
                       'B': [0, 1, 0, 1],
                       'Y': [0, 0, 1, 1]})

    temp_stdout = StringIO()
    with contextlib.redirect_stdout(temp_stdout):
        filter.fit(df)
    output = temp_stdout.getvalue().strip()

    return output


def test_verbose_output_for_top_features():
    expected_output = ("The feature importance of column 'B' (0.0000) is " +
                       "too low to end up in the 1 best features")
    tree_based_filter = TreeBasedFilter(target_column='Y', top_features=1)

    output = _capture_verbose_output_for_model(tree_based_filter)

    assert output == expected_output


def test_verbose_output_for_threshold():
    expected_output = ("The feature importance of column 'B' (0.0000) is " +
                       "below the threshold of 0.5000")
    tree_based_filter = TreeBasedFilter(target_column='Y', threshold=0.5)

    output = _capture_verbose_output_for_model(tree_based_filter)

    assert output == expected_output


if __name__ == '__main__':
    unittest.main()
