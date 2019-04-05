from typing import List  # NOQA

import numpy as np
import pandas as pd

from .abstract_transformer import AbstractTransformer


class FeatureCorrelationFilter(AbstractTransformer):
    """Remove variables above a given correlation to other feature variables.

    Args:
        max_correlation: Maximum absolute correlation. Default 0.95.
        y_col: The name of the column containing the target variable (y).
        sample_ratio: The ratio of the data to use when calculating the
            correlation. Default: 1.0.
        seed: An optional seed for sampling when using a sample_ratio < 1.0.
        verbose: Whether to print information about detected high correlation.
            Default: True.
    """
    def __init__(self,
                 max_correlation: float = 0.95,
                 target_column: str = '',
                 sample_ratio: float = 1.0,
                 seed: int = None,
                 verbose: bool = True):
        self.max_correlation = max_correlation
        self.target_column = target_column
        self.sample_ratio = sample_ratio
        self.seed = seed
        self.verbose = verbose

        self.columns_to_drop = []  # type: List[str]

    def fit(self, df: pd.DataFrame, *args, **kwargs) -> None:
        # Do not include target column
        feature_column_names = [cn for cn in df.columns
                                if cn != self.target_column]

        for i, n in enumerate(feature_column_names):
            if n in self.columns_to_drop:
                continue

            # Encode categorical columns
            if self.sample_ratio < 1.0:
                sample_size = int(len(df[n]) * self.sample_ratio)
                n_col = df[n].sample(sample_size, random_state=self.seed)
            else:
                n_col = df[n]
            column_type = n_col.dtype
            if column_type not in (np.float64, np.int64):
                n_col = n_col.astype('category').cat.codes

            for m in feature_column_names[i + 1:]:
                # Encode categorical columns
                if self.sample_ratio < 1.0:
                    m_col = df[m].sample(sample_size, random_state=self.seed)
                else:
                    m_col = df[m]
                column_type = m_col.dtype
                if column_type not in (np.float64, np.int64):
                    m_col = m_col.astype('category').cat.codes

                correlation = abs(n_col.corr(m_col))

                if correlation > self.max_correlation:
                    if self.verbose:
                        print(("The absolute correlation of column '%s' " +
                               "(%0.4f) to column '%s' is above the " +
                               "threshold of %0.4f")
                              % (n, correlation, m, self.max_correlation))
                    self.columns_to_drop.append(m)

    def transform(self, df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        return df.drop(columns=self.columns_to_drop)
