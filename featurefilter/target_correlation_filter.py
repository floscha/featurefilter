from typing import List  # NOQA

import numpy as np
import pandas as pd

from .abstract_transformer import AbstractTransformer


class TargetCorrelationFilter(AbstractTransformer):
    """Remove variables above/below a given correlation to the target variable.

    Args:
        y_col: The name of the column containing the target variable (y).
        min_correlation: Minimum absolute correlation. Default 0.01.
        max_correlation: Maximum absolute correlation. Default 0.95.
        sample_ratio: The ratio of the data to use when calculating the
            correlation. Default: 1.0.
        verbose: Whether to print information about detected high correlation.
            Default: True.
    """
    def __init__(self,
                 target_column: str,
                 min_correlation: float = 0.01,
                 max_correlation: float = 0.95,
                 sample_ratio: float = 1.0,
                 seed: int = None,
                 verbose: bool = True):
        self.target_column = target_column
        self.min_correlation = min_correlation
        self.max_correlation = max_correlation
        self.sample_ratio = sample_ratio
        self.seed = seed
        self.verbose = verbose

        self.columns_to_drop = []  # type: List[str]

    def fit(self, df: pd.DataFrame, *args, **kwargs) -> None:
        for n in df.columns:
            if n == self.target_column:
                continue

            current_column = df[n]
            column_type = current_column.dtype

            # Encode categorical columns
            if column_type not in (np.float64, np.int64):
                current_column = current_column.astype('category').cat.codes

            if self.sample_ratio < 1.0:
                sample_size = int(len(current_column) * self.sample_ratio)
                sample = df[[n, self.target_column]].sample(
                    sample_size, random_state=self.seed
                )
                correlation = sample[n].corr(sample[self.target_column])
            else:
                correlation = current_column.corr(df[self.target_column])

            correlation = abs(correlation)

            if correlation < self.min_correlation:
                if self.verbose:
                    print(("The absolute correlation of column '%s' (%0.4f) " +
                           "to the target column '%s' is below the " +
                           "threshold of %0.4f")
                          % (n, correlation, self.target_column,
                             self.min_correlation))
                self.columns_to_drop.append(n)
                continue  # No need to check for max correlation

            if correlation > self.max_correlation:
                if self.verbose:
                    print(("The absolute correlation of column '%s' (%0.4f) " +
                           "to the target column '%s' is above the " +
                           "threshold of %0.4f")
                          % (n, correlation, self.target_column,
                             self.max_correlation))
                self.columns_to_drop.append(n)

    def transform(self, df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        return df.drop(columns=self.columns_to_drop)
