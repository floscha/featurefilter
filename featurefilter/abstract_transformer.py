import pandas as pd


class AbstractTransformer:
    """An abstract class that provides an sklearn-like API."""
    def fit(self, df: pd.DataFrame) -> None:
        raise NotImplementedError()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)
