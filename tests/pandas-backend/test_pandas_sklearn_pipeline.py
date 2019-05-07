import unittest

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from featurefilter import NaFilter, VarianceFilter


def _build_test_data():
    train_df = pd.DataFrame({'A': [0, np.nan, np.nan],
                             'B': [0, 0, 0],
                             'C': [0, np.nan, 1]})
    test_df = pd.DataFrame({'A': [0, 1, 2],
                            'B': [2, 1, 0],
                            'C': [np.nan, np.nan, np.nan]})

    return train_df, test_df


def _build_test_pipeline():
    pipeline = Pipeline([
        ('na_filter', NaFilter(max_na_ratio=0.5)),
        ('variance_filter', VarianceFilter())
    ])
    return pipeline


def test_pipeline_fit_sets_correct_columns_to_drop():
    train_df, _ = _build_test_data()

    pipeline = _build_test_pipeline()
    pipeline.fit(train_df)

    assert pipeline.steps[0][1].columns_to_drop == ['A']
    assert pipeline.steps[1][1].columns_to_drop == ['B']


def test_pipeline_fit_transform_returns_correct_dataframe():
    train_df, test_df = _build_test_data()

    pipeline = _build_test_pipeline()
    train_df = pipeline.fit_transform(train_df)
    test_df = pipeline.transform(test_df)

    assert train_df.equals(pd.DataFrame({'C': [0, np.nan, 1]}))
    assert test_df.equals(pd.DataFrame({'C': [np.nan, np.nan, np.nan]}))


if __name__ == '__main__':
    unittest.main()
