import contextlib
from io import StringIO
import unittest

import pandas as pd
import pytest
from sklearn.feature_selection import (chi2, RFE, RFECV, SelectFdr, SelectFpr,
                                       SelectFromModel, SelectFwe, SelectKBest,
                                       SelectPercentile, VarianceThreshold)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from featurefilter import SklearnWrapper


def test_removing_features_with_low_variance():
    train_df = pd.DataFrame({'A': [0, 1, 2], 'B': [0, 0, 0]})
    test_df = pd.DataFrame({'A': [0, 0, 0], 'B': [0, 1, 2]})

    sklearn_selector = VarianceThreshold(threshold=0.5)
    variance_threshold = SklearnWrapper(sklearn_selector)
    train_df = variance_threshold.fit_transform(train_df)
    test_df = variance_threshold.transform(test_df)

    # Make sure column 'B' is dropped for both train and test set
    # Also, column 'A' must not be dropped for the test set even though its
    # variance in the test set is below the threshold
    assert train_df.equals(pd.DataFrame({'A': [0, 1, 2]}))
    assert test_df.equals(pd.DataFrame({'A': [0, 0, 0]}))


def test_rfe():
    train_df = pd.DataFrame({'A': [0, 0, 1, 1],
                             'B': [0, 1, 0, 1],
                             'Y': [0, 0, 1, 1]})
    expected_output = pd.DataFrame({'A': [0, 0, 1, 1],
                                    'Y': [0, 0, 1, 1]})

    model = RFE(LinearRegression(), n_features_to_select=1)
    selector = SklearnWrapper(model, target_column='Y')
    train_df = selector.fit_transform(train_df)

    assert train_df.equals(expected_output)


def test_rfecv():
    train_df = pd.DataFrame({'A': [0, 0, 1, 1],
                             'B': [0, 1, 0, 1],
                             'Y': [0, 0, 1, 1]})
    expected_output = pd.DataFrame({'A': [0, 0, 1, 1],
                                    'Y': [0, 0, 1, 1]})

    model = RFECV(LinearRegression(), min_features_to_select=1, cv=3)
    selector = SklearnWrapper(model, target_column='Y')
    train_df = selector.fit_transform(train_df)

    assert train_df.equals(expected_output)


def test_univariate_feature_selection():
    train_df = pd.DataFrame({'A': [0, 0, 1, 1],
                             'B': [0, 1, 0, 1],
                             'Y': [0, 0, 1, 1]})
    expected_output = pd.DataFrame({'A': [0, 0, 1, 1],
                                    'Y': [0, 0, 1, 1]})

    k_best_selector = SelectKBest(chi2, k=1)
    selector = SklearnWrapper(k_best_selector, target_column='Y')
    train_df = selector.fit_transform(train_df)

    assert train_df.equals(expected_output)


def test_missing_target_column():
    k_best_selector = SelectKBest(chi2, k=1)

    with pytest.raises(ValueError) as excinfo:
        SklearnWrapper(k_best_selector)

    assert str(excinfo.value) == "A target columns must be set for SelectKBest"


def test_glm_based_feature_selection():
    train_df = pd.DataFrame({'A': [0, 0, 1, 1],
                             'B': [0, 1, 0, 1],
                             'Y': [0, 0, 1, 1]})
    expected_output = pd.DataFrame({'A': [0, 0, 1, 1],
                                    'Y': [0, 0, 1, 1]})

    model = SelectFromModel(LinearRegression(), threshold=0.5)
    selector = SklearnWrapper(model, target_column='Y')
    train_df = selector.fit_transform(train_df)

    assert train_df.equals(expected_output)


def test_prefit_model():
    train_df = pd.DataFrame({'A': [0, 0, 1, 1],
                             'B': [0, 1, 0, 1],
                             'Y': [0, 0, 1, 1]})
    expected_output = pd.DataFrame({'A': [0, 0, 1, 1],
                                    'Y': [0, 0, 1, 1]})

    linear_regression = LinearRegression()
    linear_regression.fit(train_df[['A', 'B']], train_df['Y'])
    model = SelectFromModel(linear_regression, prefit=True, threshold=0.5)
    selector = SklearnWrapper(model, target_column='Y')
    train_df = selector.transform(train_df)

    assert train_df.equals(expected_output)


def test_tree_based_feature_selection():
    train_df = pd.DataFrame({'A': [0, 0, 1, 1],
                             'B': [0, 1, 0, 1],
                             'Y': [0, 0, 1, 1]})
    expected_output = pd.DataFrame({'A': [0, 0, 1, 1],
                                    'Y': [0, 0, 1, 1]})

    model = SelectFromModel(DecisionTreeRegressor(), threshold=0.5)
    selector = SklearnWrapper(model, target_column='Y')
    train_df = selector.fit_transform(train_df)

    assert train_df.equals(expected_output)


def _capture_verbose_output_for_model(model, use_supervised_df):
    if use_supervised_df:
        df = pd.DataFrame({'A': [0, 0, 1, 1],
                           'B': [0, 1, 0, 1],
                           'Y': [0, 0, 1, 1]})
        target_column = 'Y'
    else:
        df = pd.DataFrame({'A': [0, 1, 2], 'B': [0, 0, 0]})
        target_column = None

    filter = SklearnWrapper(model, target_column)
    filter.fit(df)

    temp_stdout = StringIO()
    with contextlib.redirect_stdout(temp_stdout):
        filter.print_columns_to_drop()
    output = temp_stdout.getvalue().strip()

    return output


def test_verbose_output_for_variance_threshold():
    expected_output = ("The variance of column 'B' (0.0000) is below the " +
                       "threshold of 0.5000")
    model = VarianceThreshold(threshold=0.5)

    output = _capture_verbose_output_for_model(model, use_supervised_df=False)

    assert output == expected_output


def test_verbose_output_for_select_from_model():
    expected_output = ("The feature importance of column 'B' (0.0000) is " +
                       "below the threshold of 0.5000")

    model = SelectFromModel(LinearRegression(), threshold=0.5)

    output = _capture_verbose_output_for_model(model, use_supervised_df=True)

    assert output == expected_output


def test_verbose_output_for_select_percentile():
    expected_output = ("The feature importance of column 'B' (0.0000) is " +
                       "out of the 10% of features to keep")

    model = SelectPercentile(chi2, percentile=10)

    output = _capture_verbose_output_for_model(model, use_supervised_df=True)

    assert output == expected_output


def test_verbose_output_for_select_select_fpr():
    expected_output = ("The p-value of column 'B' (1.0000) is above the " +
                       "specified alpha of 0.5000")

    model = SelectFpr(chi2, alpha=0.5)

    output = _capture_verbose_output_for_model(model, use_supervised_df=True)

    assert output == expected_output


def test_verbose_output_for_select_select_fdr():
    expected_output = ("The p-value of column 'B' (1.0000) is above the " +
                       "specified alpha of 0.5000")

    model = SelectFdr(chi2, alpha=0.5)

    output = _capture_verbose_output_for_model(model, use_supervised_df=True)

    assert output == expected_output


def test_verbose_output_for_select_select_fwe():
    expected_output = ("The p-value of column 'B' (1.0000) is above the " +
                       "specified alpha of 0.5000")

    model = SelectFwe(chi2, alpha=0.5)

    output = _capture_verbose_output_for_model(model, use_supervised_df=True)

    assert output == expected_output


def test_verbose_output_for_select_select_k_best():
    expected_output = ("The feature importance of column 'B' (0.0000) is " +
                       "too low to end up in the 1 best features")

    model = SelectKBest(chi2, k=1)

    output = _capture_verbose_output_for_model(model, use_supervised_df=True)

    assert output == expected_output


def test_verbose_output_for_select_rfe():
    expected_output = ("The feature importance of column 'B' is " +
                       "too low to end up in the 1 best features")

    model = RFE(LinearRegression(), n_features_to_select=1)

    output = _capture_verbose_output_for_model(model, use_supervised_df=True)

    assert output == expected_output


def test_verbose_output_for_select_rfecv():
    expected_output = ("The feature importance of column 'B' is " +
                       "too low to end up in the 1 best features")

    model = RFECV(LinearRegression(), min_features_to_select=1, cv=3)

    output = _capture_verbose_output_for_model(model, use_supervised_df=True)

    assert output == expected_output


if __name__ == '__main__':
    unittest.main()
