import contextlib
from io import StringIO
import unittest

import pandas as pd
import pytest
from sklearn.linear_model import (ElasticNet, Lasso, LinearRegression,
                                  LogisticRegression, Ridge)

from featurefilter import GLMFilter


def test_linear_regression_is_default_continuous_model():
    glm_filter = GLMFilter(target_column='')

    model = glm_filter._model
    assert isinstance(model, LinearRegression)


def test_logistic_regression_is_default_categorical_model():
    glm_filter = GLMFilter(target_column='', categorical_target=True)

    model = glm_filter._model
    assert isinstance(model, LogisticRegression)


def test_invalid_model_type():
    invalid_model_type = 'UNK'
    with pytest.raises(ValueError) as excinfo:
        GLMFilter(target_column='', model_type=invalid_model_type)

    assert str(excinfo.value).startswith("Model '%s' not available."
                                         % invalid_model_type)


def test_continuous_logistic_regression_raises_exception():
    invalid_model_type = 'LogisticRegression'
    with pytest.raises(ValueError) as excinfo:
        GLMFilter(target_column='', model_type=invalid_model_type)

    assert str(excinfo.value).startswith("%s cannot be used"
                                         % invalid_model_type)


def test_categorical_linear_regression_raises_exception():
    invalid_model_type = 'LinearRegression'
    with pytest.raises(ValueError) as excinfo:
        GLMFilter(target_column='', categorical_target=True,
                  model_type=invalid_model_type)

    assert str(excinfo.value).startswith("%s cannot be used"
                                         % invalid_model_type)


def test_ridge_model_type():
    glm_filter = GLMFilter(target_column='', model_type='Ridge')

    model = glm_filter._model
    assert isinstance(model, Ridge)


def test_lasso_model_type():
    glm_filter = GLMFilter(target_column='', model_type='Lasso')

    model = glm_filter._model
    assert isinstance(model, Lasso)


def test_elastic_net_model_type():
    glm_filter = GLMFilter(target_column='', model_type='ElasticNet')

    model = glm_filter._model
    assert isinstance(model, ElasticNet)


def test_fit_sets_correct_columns_to_drop_with_continuous_target():
    train_df = pd.DataFrame({'A': [0, 0, 1, 1],
                             'B': [0, 1, 0, 1],
                             'Y': [0, 0, 1, 1]})

    glm_filter = GLMFilter(target_column='Y', top_features=1)
    glm_filter.fit(train_df)

    assert glm_filter.columns_to_drop == ['B']


def test_fit_sets_correct_columns_to_drop_with_categorical_target():
    train_df = pd.DataFrame({'A': [0, 0, 1, 1],
                             'B': [0, 1, 0, 1],
                             'Y': ['a', 'a', 'b', 'b']})

    glm_filter = GLMFilter(target_column='Y', categorical_target=True,
                           top_features=1)
    glm_filter.fit(train_df)

    assert glm_filter.columns_to_drop == ['B']


def test_setting_model_parameters():
    model_parameters = {'alpha': 0.5}

    glm_filter = GLMFilter(target_column='Y',
                           model_type='Ridge',
                           model_parameters=model_parameters)

    assert glm_filter._model.alpha == 0.5


def test_fit_transform_removes_nothing_by_default():
    train_df = pd.DataFrame({'A': [0, 0, 1, 1],
                             'B': [0, 1, 0, 1],
                             'Y': [0, 0, 1, 1]})
    expected_output = pd.DataFrame({'A': [0, 0, 1, 1],
                                    'B': [0, 1, 0, 1],
                                    'Y': [0, 0, 1, 1]})

    glm_filter = GLMFilter(target_column='Y')
    train_df = glm_filter.fit_transform(train_df)

    assert train_df.equals(expected_output)


def test_fit_transform_top_features():
    train_df = pd.DataFrame({'A': [0, 0, 1, 1],
                             'B': [0, 1, 0, 1],
                             'Y': [0, 0, 1, 1]})
    expected_output = pd.DataFrame({'A': [0, 0, 1, 1],
                                    'Y': [0, 0, 1, 1]})

    glm_filter = GLMFilter(target_column='Y', top_features=1)
    train_df = glm_filter.fit_transform(train_df)

    assert train_df.equals(expected_output)


def test_fit_transform_threshold():
    train_df = pd.DataFrame({'A': [0, 0, 1, 1],
                             'B': [0, 1, 0, 1],
                             'Y': [0, 0, 1, 1]})
    expected_output = pd.DataFrame({'A': [0, 0, 1, 1],
                                    'Y': [0, 0, 1, 1]})

    glm_filter = GLMFilter(target_column='Y', threshold=0.5)
    train_df = glm_filter.fit_transform(train_df)

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
    glm_filter = GLMFilter(target_column='Y', top_features=1)

    output = _capture_verbose_output_for_model(glm_filter)

    assert output == expected_output


def test_verbose_output_for_threshold():
    expected_output = ("The feature importance of column 'B' (0.0000) is " +
                       "below the threshold of 0.5000")
    glm_filter = GLMFilter(target_column='Y', threshold=0.5)

    output = _capture_verbose_output_for_model(glm_filter)

    assert output == expected_output


def test_sparse_data():
    # Create a sparse DataFrame with a correlation of -1
    sdf = pd.DataFrame({'A': [0] * 10 + [1] * 10,
                        'B': [0, 1] * 10,
                        'Y': [0] * 10 + [1] * 10}).to_sparse()
    glm_filter = GLMFilter(target_column='Y', top_features=1)

    glm_filter.fit(sdf)

    assert glm_filter.columns_to_drop == ['B']


if __name__ == '__main__':
    unittest.main()
