from typing import List, Union  # NOQA

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin  # NOQA
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor # NOQA
from sklearn.ensemble import (GradientBoostingClassifier, # NOQA
                              GradientBoostingRegressor,
                              RandomForestClassifier,
                              RandomForestRegressor)

from .abstract_transformer import AbstractTransformer


class TreeBasedFilter(AbstractTransformer):
    def __init__(self,
                 target_column: str,
                 categorical_target: bool = False,
                 top_features: int = None,
                 relative_treshold: float = None,
                 model_type: str = 'DecisionTree',
                 model_parameters=None,
                 verbose: bool = True):
        self.target_column = target_column
        self.top_features = top_features
        self.relative_treshold = relative_treshold
        self.verbose = verbose

        self.columns_to_drop = []  # type: List[str]

        available_models = ('DecisionTree', 'GradientBoosting', 'RandomForest')
        if model_type not in available_models:
            raise ValueError(("Model '%s' not available. Please choose one " +
                              "of the following instead: %s")
                             % (model_type, available_models))
        model_parameters = model_parameters if model_parameters else {}
        model_class_name = model_type + ('Classifier' if categorical_target
                                         else 'Regressor')
        self._model = globals()[model_class_name](**model_parameters)

    def fit(self, df: pd.DataFrame) -> None:
        feature_column_names = np.array(
            [cn for cn in df.columns if cn != self.target_column]
        )
        self._model.fit(df[feature_column_names],
                        df[self.target_column])

        feature_importances = self._model.feature_importances_
        if self.verbose:
            for cn, fi in zip(feature_column_names, feature_importances):
                print(cn, fi)

        feature_names = feature_column_names[np.argsort(feature_importances)]
        print(feature_names)
        top_features_names = (feature_names if not self.top_features
                              else feature_names[:-self.top_features])
        self.columns_to_drop = list(top_features_names)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=self.columns_to_drop)
