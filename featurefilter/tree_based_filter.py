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
                 threshold: float = None,
                 model_type: str = 'DecisionTree',
                 model_parameters=None,
                 verbose: bool = True):
        self.target_column = target_column
        self.top_features = top_features
        self.threshold = threshold
        self.verbose = verbose

        self.columns_to_drop = []  # type: List[str]

        available_models = ('DecisionTree', 'GradientBoosting', 'RandomForest')
        if model_type not in available_models:
            raise ValueError(("Model '%s' not available. Please choose one " +
                              "of the following instead: %s")
                             % (model_type, available_models))
        model_class_name = model_type + ('Classifier' if categorical_target
                                         else 'Regressor')
        model_parameters = model_parameters if model_parameters else {}
        self._model = globals()[model_class_name](**model_parameters)

    def fit(self, df: pd.DataFrame, *args, **kwargs) -> None:
        feature_column_names = np.array(
            [cn for cn in df.columns if cn != self.target_column]
        )
        self._model.fit(df[feature_column_names],
                        df[self.target_column])

        feature_importances = self._model.feature_importances_

        if self.top_features:
            top_feature_indices = np.argsort(feature_importances)[::-1]
            feature_names = feature_column_names[top_feature_indices]
            top_feature_importances = feature_importances[top_feature_indices]
            for i, (cn, fi) in enumerate(zip(feature_names,
                                             top_feature_importances)):
                if i < self.top_features:
                    continue
                self.columns_to_drop.append(cn)
                if self.verbose:
                    print(("The feature importance of column '%s' (%0.4f) " +
                           "is too low to end up in the %d best features")
                          % (cn, fi, self.top_features))

        if self.threshold is not None:
            for cn, fi in zip(feature_column_names, feature_importances):
                if fi < self.threshold:
                    self.columns_to_drop.append(cn)
                    if self.verbose:
                        print(("The feature importance of column '%s' " +
                               "(%0.4f) is below the threshold of %0.4f")
                              % (cn, fi, self.threshold))

    def transform(self, df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        return df.drop(columns=self.columns_to_drop)
