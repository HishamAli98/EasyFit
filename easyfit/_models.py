from abc import ABC
from typing import Dict

import pandas as pd
import tqdm


class _EasyModel(ABC):
    """Base model for EasyRegressor and EasyClassifier

    Notes
    -----
    Abstract Class, do not use for training
    Use subclasses, i.e. EasyRegressor, EasyClassifier

    """
    METRICS = {}

    def __new__(cls, *args, **kwargs):
        if cls is _EasyModel:
            raise TypeError(f"Cannot create object of type '{cls.__name__}',"
                            f"use 'EasyRegressor' or 'EasyClassifier'.")
        return object.__new__(cls)

    def __init__(self,
                 default_models: Dict,
                 models_dict: Dict = None,
                 include_defaults: bool = True,
                 ):
        if models_dict is None:
            models_dict = {}
        assert isinstance(
            models_dict, dict), "models_dict must be of type dict"
        assert models_dict or include_defaults, \
            "must supply models_dict or set include_defaults to True"
        if include_defaults:
            self.models_dict = {**default_models, **models_dict}
        for model_key, model in self.models_dict.items():
            for attr in ("fit", "predict"):
                assert hasattr(
                    model, attr), f"model {model_key} must have {attr} method"
        self._models = {
            model_key: model_class
            if self._is_instantiated(model_class)
            else model_class()
            for model_key, model_class in self.models_dict.items()
        }

    def fit(self, X, y):
        for model in tqdm.tqdm(self._models.values()):
            model.fit(X, y)

    def predict(self, X):
        return {model_key: model.predict(X)
                for model_key, model in self._models.items()}

    def score(self, X, y, as_df=False, sorted=True):
        result = {model_key: (model.score(X, y)
                  if hasattr(model, 'score') else 'NA')
                  for model_key, model in self._models.items()}
        if sorted:
            result = dict(
                sorted(result.items(), key=lambda x: x[1], reverse=True)
            )
        if as_df:
            result = pd.DataFrame(result.items(), columns=["Model", "Score"])
        return result

    def evaluate(self, X, y, as_df=False,
                 model_first=True, from_preds=False):
        results = {}
        models_preds = X if from_preds else self.predict(X)
        if model_first:
            for model, y_preds in models_preds.items():
                results[model] = {metric: func(y, y_preds)
                                  for metric, func in self.METRICS.items()}

        else:
            for metric, func in self.METRICS.items():
                results[metric] = {model: func(y, y_preds) for model,
                                   y_preds in models_preds.items()}

        if as_df:
            results = pd.DataFrame.from_dict(results, orient="index")
        return results

    def get_model(self, model_key):
        return self._models.get(model_key, None)

    @staticmethod
    def _is_instantiated(object_: object) -> bool:
        return not isinstance(object_, type)
