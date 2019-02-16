from typing import List  # NOQA

import numpy as np
import pandas as pd
from sklearn.linear_model import (ElasticNet, Lasso, LinearRegression,  # NOQA
                                  LogisticRegression, Ridge)

from .abstract_transformer import AbstractTransformer


class GLMFilter(AbstractTransformer):
    def __init__(self,
                 target_column: str,
                 categorical_target: bool = False,
                 top_features: int = None,
                 relative_treshold: float = None,
                 model_type: str = None,
                 model_parameters=None,
                 verbose: bool = True):
        self.target_column = target_column
        self.categorical_target = categorical_target
        self.top_features = top_features
        self.relative_treshold = relative_treshold
        self.verbose = verbose

        self.columns_to_drop = []  # type: List[str]

        if model_type is None:
            if categorical_target:
                model_type = 'LogisticRegression'
            else:
                model_type = 'LinearRegression'

        available_models = ('ElasticNet', 'Lasso', 'LinearRegression',
                            'LogisticRegression', 'Ridge')
        if model_type not in available_models:
            raise ValueError(("Model '%s' not available. Please choose one " +
                              "of the following instead: %s")
                             % (model_type, available_models))

        if not categorical_target and model_type == 'LogisticRegression':
            raise ValueError('%s cannot be used for continuous targets'
                             % model_type)
        elif categorical_target and model_type != 'LogisticRegression':
            raise ValueError('%s cannot be used for categorical targets'
                             % model_type)

        model_parameters = model_parameters if model_parameters else {}
        # Use lbfgs as default solver for logistic regression
        if model_type == 'LogisticRegression':
            model_parameters['solver'] = model_parameters.get('solver',
                                                              'lbfgs')
        self._model = globals()[model_type](**model_parameters)

    def fit(self, df: pd.DataFrame, *args, **kwargs) -> None:
        feature_column_names = np.array(
            [cn for cn in df.columns if cn != self.target_column]
        )
        self._model.fit(df[feature_column_names],
                        df[self.target_column])

        feature_importances = (self._model.coef_ if not self.categorical_target
                               else self._model.coef_[0])
        # Take absolute feature importances to treat positive/negative
        # coefficients the same
        feature_importances = abs(feature_importances)
        # Use normalized importances on a scale from 0 to 1
        feature_importances = feature_importances / max(feature_importances)
        if self.verbose:
            print("Relative feature importances:")
            for cn, fi in zip(feature_column_names, feature_importances):
                print('  ', cn, '%0.4f' % fi)

        feature_names = feature_column_names[np.argsort(feature_importances)]
        if self.top_features:
            self.columns_to_drop = feature_names[:-self.top_features]

        if self.relative_treshold is not None:
            self.columns_to_drop.extend([
                cn for cn in
                feature_column_names[feature_importances
                                     < self.relative_treshold]
                if cn not in self.columns_to_drop
            ])

    def transform(self, df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        return df.drop(columns=self.columns_to_drop)
