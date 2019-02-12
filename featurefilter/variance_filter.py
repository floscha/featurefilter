from typing import List  # NOQA

import numpy as np
import pandas as pd

from .abstract_transformer import AbstractTransformer


class VarianceFilter(AbstractTransformer):
    """Remove variables below a certain variance threshold.

    For continuous variables:
        Simple variance like used by TransmogrifAI:
        https://github.com/salesforce/TransmogrifAI/blob/master/core/src/main/
            scala/com/salesforce/op/stages/impl/preparators/SanityChecker.scala
    For categorical variables:
        Near-zero-variance:
        http://topepo.github.io/caret/pre-processing.html#nzv
        The default parameters for `freq_cut` and `uniq_cut` are taken from:
        https://github.com/topepo/caret/blob/
            6546939345fe10649cefcbfee55d58fb682bc902/pkg/caret/R/nearZeroVar.R#L90
    """
    def __init__(self,
                 min_variance: float = 0.00001,
                 frequency_cut: float = 95/5,
                 unique_cut: float = 10.0,
                 sample_ratio: float = 1.0,
                 seed: int = None,
                 verbose: bool = True):
        self.min_variance = min_variance
        self.frequency_cut = frequency_cut
        self.unique_cut = unique_cut
        self.sample_ratio = sample_ratio
        self.seed = seed
        self.verbose = verbose

        self.columns_to_drop = []  # type: List[str]

    @staticmethod
    def _get_freq_ratio(column: pd.Series) -> float:
        """Compute frequency ratio."""
        value_counts = column.value_counts(normalize=True)
        if len(value_counts) == 1:
            return float('inf')
        most_common_freq, second_most_common_freq = value_counts[:2]
        frequency_ratio = most_common_freq - second_most_common_freq
        return frequency_ratio

    @staticmethod
    def _get_percentage_of_unique_values(column: pd.Series) -> float:
        """Compute percentage of unique values."""
        return len(column.unique()) / len(column) * 100

    def fit(self, df: pd.DataFrame, *args, **kwargs) -> None:
        for n in df.columns:
            current_column = df[n]
            column_type = current_column.dtype

            # Continuous columns
            if column_type in (np.float64, np.int64):
                if self.sample_ratio < 1.0:
                    sample_size = int(len(current_column) * self.sample_ratio)
                    n_variance = (current_column
                                  .sample(sample_size, random_state=self.seed)
                                  .var())
                else:
                    n_variance = current_column.var()
                if n_variance < self.min_variance:
                    if self.verbose:
                        print(("The variance of column '%s' (%0.4f) is " +
                               "below the threshold of %0.4f")
                              % (n, n_variance, self.min_variance))
                    self.columns_to_drop.append(n)
            # Categorical columns
            elif column_type == pd.Categorical:
                is_above_frequency_cut = (
                    self._get_freq_ratio(current_column)
                    > self.frequency_cut
                )
                is_below_unique_cut = (
                    self._get_percentage_of_unique_values(current_column)
                    < self.unique_cut
                )
                if is_above_frequency_cut and is_below_unique_cut:
                    if self.verbose:
                        print(("Column '%s' is above the frequency cut of " +
                               "%0.4f and below the unique cut of %0.4f")
                              % (n, self.frequency_cut, self.unique_cut))
                    self.columns_to_drop.append(n)

    def transform(self, df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        return df.drop(columns=self.columns_to_drop)
