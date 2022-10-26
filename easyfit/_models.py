from abc import ABC
from typing import Dict

import pandas as pd
import tqdm
from sklearn.base import BaseEstimator

ModelsDict = Dict[str, BaseEstimator]


class EasyModel(ABC):
    METRICS = {}

    def __new__(cls, *args, **kwargs):
        if cls is EasyModel:
            raise TypeError(f"Cannot create object of type '{cls.__name__}',"
                            f"use 'EasyRegressor' or 'EasyClassifier'.")
        return object.__new__(cls)

    def __init__(self,
                 default_models,
                 models_dict: ModelsDict = None,
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
        self._models = {model_key: model_class()
                        for model_key, model_class in self.models_dict.items()}

    def fit(self, X, y):
        for model in tqdm.tqdm(self._models.values()):
            model.fit(X, y)

    def predict(self, X):
        return {model_key: model.predict(X)
                for model_key, model in self._models.items()}

    def score(self, X, y, as_df=False):
        result = {model_key: model.score(X, y)
                  for model_key, model in self._models.items()}
        if as_df:
            result = pd.DataFrame(result.items(), columns=["Model", "Score"])
        return result

    def evaluate(self, inputs, targets, as_df=False,
                 model_first=True, from_preds=False):
        results = {}
        models_preds = inputs if from_preds else self.predict(inputs)
        if model_first:
            for model, y_preds in models_preds.items():
                results[model] = {}
                for metric, func in self.METRICS.items():
                    results[model][metric] = func(targets, y_preds)
        else:
            for metric, func in self.METRICS.items():
                results[metric] = {}
                for model, y_preds in models_preds.items():
                    results[metric][model] = func(targets, y_preds)

        if as_df:
            results = pd.DataFrame.from_dict(results, orient="index")
        return results

    def get_model(self, model_key):
        return self._models.get(model_key, None)

    def __getitem__(self, model_key):
        return self._models[model_key]
