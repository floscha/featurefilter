from typing import List  # NOQA

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection.base import SelectorMixin
from sklearn.feature_selection.from_model import _get_feature_importances
from sklearn.feature_selection.univariate_selection import _BaseFilter

from .abstract_transformer import AbstractTransformer


class SklearnWrapper(AbstractTransformer):
    """Remove variables using sklearn's feature_selection module."""
    def __init__(self,
                 selector: SelectorMixin,
                 target_column: str = None,
                 sample_ratio: float = 1.0,
                 seed: int = None,
                 verbose: bool = True):
        self.selector = selector

        if isinstance(selector, _BaseFilter) and not target_column:
            raise ValueError("A target columns must be set for %s"
                             % selector.__class__.__name__)

        self.target_column = target_column
        self.sample_ratio = sample_ratio
        self.seed = seed
        self.verbose = verbose

        self.columns_to_drop = []  # type: List[str]
        self._indices_to_drop = []  # type: List[str]

    def print_columns_to_drop(self):
        """Print information if and why a column was dropped."""
        for ci, cn in zip(self._indices_to_drop, self.columns_to_drop):
            selector_attributes = self.selector.__dict__
            if 'threshold' in selector_attributes:
                # VarianceThreshold
                if self.selector.__class__.__name__ == 'VarianceThreshold':
                    print(("The variance of column '%s' (%0.4f) is " +
                           "below the threshold of %0.4f")
                          % (cn, self.selector.variances_[ci],
                             self.selector.threshold))
                # SelectFromModel
                if self.selector.__class__.__name__ == 'SelectFromModel':
                    # The fitted estimator ends with an underscore
                    estimator = self.selector.estimator_
                    print(("The feature importance of column '%s' " +
                           "(%0.4f) is below the threshold of %0.4f")
                          % (cn, _get_feature_importances(estimator)[ci],
                             self.selector.threshold))
            elif 'percentile' in selector_attributes:
                # SelectPercentile
                print(("The feature importance of column '%s' (%0.4f) is " +
                       "out of the %d%% of features to keep")
                      % (cn, self.selector.scores_[ci],
                         self.selector.percentile))
            elif 'alpha' in selector_attributes:
                # SelectFpr, SelectFdr, SelectFwe
                print(("The p-value of column '%s' (%0.4f) is above the " +
                       "specified alpha of %0.4f")
                      % (cn, self.selector.pvalues_[ci], self.selector.alpha))
            elif 'k' in selector_attributes:
                # SelectKBest
                print(("The feature importance of column '%s' (%0.4f) is " +
                       "too low to end up in the %d best features")
                      % (cn, self.selector.scores_[ci], self.selector.k))
            elif 'n_features_to_select' in selector_attributes:
                # RFE
                estimator = self.selector.estimator_
                print(("The feature importance of column '%s' is " +
                       "too low to end up in the %d best features")
                      % (cn, self.selector.n_features_to_select))
            elif 'min_features_to_select' in selector_attributes:
                # RFECV
                estimator = self.selector.estimator_
                print(("The feature importance of column '%s' is " +
                       "too low to end up in the %d best features")
                      % (cn, self.selector.min_features_to_select))

    def fit(self, df: pd.DataFrame, *args, **kwargs) -> None:
        column_names = df.columns
        if self.target_column:
            column_names = column_names.drop(self.target_column)
            self.selector.fit(df[column_names], df[self.target_column])
        else:
            self.selector.fit(df)
        support_mask = self.selector.get_support()
        # Inverse support_mask since it contains the columns to keep
        self.columns_to_drop = list(column_names[~support_mask])
        self._indices_to_drop = np.arange(len(column_names))[~support_mask]
        if self.verbose:
            self.print_columns_to_drop()

    def transform(self, df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        if (isinstance(self.selector, SelectFromModel)
                and self.selector.prefit):
            column_names = df.columns.drop(self.target_column)
            support_mask = self.selector._get_support_mask()
            return df.drop(columns=column_names[~support_mask])

        return df.drop(columns=self.columns_to_drop)
