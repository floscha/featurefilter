import unittest

import pandas as pd
import pytest
from sklearn import feature_selection as sklearn_feature_selection
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from featurefilter import SklearnWrapper


def test_removing_features_with_low_variance():
    train_df = pd.DataFrame({'A': [0, 1, 2], 'B': [0, 0, 0]})
    test_df = pd.DataFrame({'A': [0, 0, 0], 'B': [0, 1, 2]})

    sklearn_selector = sklearn_feature_selection.VarianceThreshold(
        threshold=0.5
    )
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

    model = sklearn_feature_selection.RFE(LinearRegression(),
                                          n_features_to_select=1)
    selector = SklearnWrapper(model, target_column='Y')
    train_df = selector.fit_transform(train_df)

    assert train_df.equals(expected_output)


def test_rfecv():
    train_df = pd.DataFrame({'A': [0, 0, 1, 1],
                             'B': [0, 1, 0, 1],
                             'Y': [0, 0, 1, 1]})
    expected_output = pd.DataFrame({'A': [0, 0, 1, 1],
                                    'Y': [0, 0, 1, 1]})

    model = sklearn_feature_selection.RFECV(LinearRegression(),
                                            min_features_to_select=1,
                                            cv=3)
    selector = SklearnWrapper(model, target_column='Y')
    train_df = selector.fit_transform(train_df)

    assert train_df.equals(expected_output)


def test_univariate_feature_selection():
    train_df = pd.DataFrame({'A': [0, 0, 1, 1],
                             'B': [0, 1, 0, 1],
                             'Y': [0, 0, 1, 1]})
    expected_output = pd.DataFrame({'A': [0, 0, 1, 1],
                                    'Y': [0, 0, 1, 1]})

    k_best_selector = sklearn_feature_selection.SelectKBest(
        sklearn_feature_selection.chi2,
        k=1
    )
    selector = SklearnWrapper(k_best_selector, target_column='Y')
    train_df = selector.fit_transform(train_df)

    assert train_df.equals(expected_output)


def test_missing_target_column():
    k_best_selector = sklearn_feature_selection.SelectKBest(
        sklearn_feature_selection.chi2,
        k=1
    )

    with pytest.raises(ValueError) as excinfo:
        SklearnWrapper(k_best_selector)

    assert str(excinfo.value) == "A target columns must be set for SelectKBest"


def test_glm_based_feature_selection():
    train_df = pd.DataFrame({'A': [0, 0, 1, 1],
                             'B': [0, 1, 0, 1],
                             'Y': [0, 0, 1, 1]})
    expected_output = pd.DataFrame({'A': [0, 0, 1, 1],
                                    'Y': [0, 0, 1, 1]})

    model = sklearn_feature_selection.SelectFromModel(LinearRegression(),
                                                      threshold=0.5)
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
    model = sklearn_feature_selection.SelectFromModel(linear_regression,
                                                      prefit=True,
                                                      threshold=0.5)
    selector = SklearnWrapper(model, target_column='Y')
    train_df = selector.transform(train_df)

    assert train_df.equals(expected_output)


def test_tree_based_feature_selection():
    train_df = pd.DataFrame({'A': [0, 0, 1, 1],
                             'B': [0, 1, 0, 1],
                             'Y': [0, 0, 1, 1]})
    expected_output = pd.DataFrame({'A': [0, 0, 1, 1],
                                    'Y': [0, 0, 1, 1]})

    model = sklearn_feature_selection.SelectFromModel(DecisionTreeRegressor(),
                                                      threshold=0.5)
    selector = SklearnWrapper(model, target_column='Y')
    train_df = selector.fit_transform(train_df)

    assert train_df.equals(expected_output)


if __name__ == '__main__':
    unittest.main()
