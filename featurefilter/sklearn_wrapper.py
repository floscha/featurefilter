from typing import List  # NOQA

import pandas as pd
from sklearn import feature_selection as sklearn_feature_selection
from sklearn.feature_selection.univariate_selection import _BaseFilter

from .abstract_transformer import AbstractTransformer


class SklearnWrapper(AbstractTransformer):
    """Remove variables using sklearn's feature_selection module."""
    def __init__(self,
                 filter: sklearn_feature_selection.base.SelectorMixin,
                 target_column: str = None,
                 sample_ratio: float = 1.0,
                 seed: int = None,
                 verbose: bool = True):
        self.filter = filter

        if isinstance(filter, _BaseFilter) and not target_column:
            raise ValueError("A target columns must be set for %s"
                             % filter.__class__.__name__)

        self.target_column = target_column
        self.sample_ratio = sample_ratio
        self.seed = seed
        self.verbose = verbose

        self.columns_to_drop = []  # type: List[str]

    def fit(self, df: pd.DataFrame, *args, **kwargs) -> None:
        columns_names = df.columns
        if self.target_column:
            columns_names = columns_names.drop(self.target_column)
            self.filter.fit(df[columns_names],
                            df[self.target_column])
        else:
            self.filter.fit(df)
        support_mask = self.filter._get_support_mask()
        # Inverse support_mask since it contains the columns to keep
        self.columns_to_drop = columns_names[~support_mask]

    def transform(self, df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        return df.drop(columns=self.columns_to_drop)
