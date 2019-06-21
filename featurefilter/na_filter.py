from typing import List  # NOQA

import pandas as pd

from .abstract_transformer import AbstractTransformer


class NaFilter(AbstractTransformer):
    """Remove variables above a certain NA ratio."""
    def __init__(self,
                 max_na_ratio: float = 0.1,
                 sample_ratio: float = 1.0,
                 seed: int = None,
                 verbose: bool = True):
        self.max_na_ratio = max_na_ratio
        self.sample_ratio = sample_ratio
        self.seed = seed
        self.verbose = verbose

        self.columns_to_drop = []  # type: List[str]

    def fit(self, df: pd.DataFrame, *args, **kwargs) -> None:
        if self.sample_ratio < 1.0:
            df = df.sample(frac=self.sample_ratio, random_state=self.seed)

        na_ratios = df.isna().mean()

        for i, n in enumerate(df.columns):
            current_na_ratio = na_ratios[i]
            if current_na_ratio > self.max_na_ratio:
                if self.verbose:
                    print(("The NA ratio of column '%s' (%0.4f) is above " +
                           "the threshold of %0.4f")
                          % (n, current_na_ratio, self.max_na_ratio))
                self.columns_to_drop.append(n)

    def transform(self, df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        return df.drop(columns=self.columns_to_drop)
