from abc import ABC
from typing import Dict

import pandas as pd
import tqdm


class EasyModel(ABC):
    """Base model for EasyRegressor and EasyClassifier

    Parameters
    ----------
    default_models : Dictionary of default models

    models_dict : Dictionary of additional models, either classes or instances
                  e.g. models_dict = {'LinearRegression': LinearRegression}
                       models_dict = {'LinearRegression': LinearRegression()}
        (Default value = None)

    models_dict : Include default_models in trained models
        (Default value = True)

    Notes
    -----
    Abstract Class, do not use for training
    Use subclasses, i.e. EasyRegressor, EasyClassifier

    """
    METRICS = {}

    def __new__(cls, *args, **kwargs):
        if cls is EasyModel:
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
        self._models = {model_key: (model_class()
                        if self._is_instantiated(model_class) else model_class)
                        for model_key, model_class in self.models_dict.items()}

    def fit(self, X, y):
        """
        Fit models in self._models on features X with targets y
        Calls method fit for each model in self._models

        Parameters
        ----------
        X : array of features

        y : array of targets


        Returns
        -------
        None

        """
        for model in tqdm.tqdm(self._models.values()):
            model.fit(X, y)

    def predict(self, X):
        """
        Make predictions for features in X
        Call predict method for each model in self._models

        Parameters
        ----------
        X : array of features


        Returns
        -------
        Dictionary with same keys in self._models and predictions for each
        model of features in X

        """
        return {model_key: model.predict(X)
                for model_key, model in self._models.items()}

    def score(self, X, y, as_df=False):
        """
        Calculate score for each model in self._models
        Calls score method for each model in self._models

        Parameters
        ----------
        X : array of features

        y : array of targets

        as_df : if True: return score results in pd.DataFrame
                if False: return score results in dictionary
            (Default value = False)

        Returns
        -------
        If as_df is True: Dictionary with same keys in self._models and score
                          for each model of features in X
        If as_df is False: pd.DataFrame with models keys in rows
                           and score in column
        """
        result = {model_key: (model.score(X, y)
                  if hasattr(model, 'score') else 'NA')
                  for model_key, model in self._models.items()}
        if as_df:
            result = pd.DataFrame(result.items(), columns=["Model", "Score"])
        return result

    def evaluate(self, inputs, targets, as_df=False,
                 model_first=True, from_preds=False):
        """

        Parameters
        ----------
        inputs :

        targets :

        as_df :
            (Default value = False)
        model_first :
            (Default value = True)
        from_preds :
            (Default value = False)

        Returns
        -------
        If as_df is True: Dictionary with same keys in self._models and
                          dictionary of scores for each model as value
        If as_df is False: pd.DataFrame with models keys in rows
                           and metrics in columns

        """
        results = {}
        models_preds = inputs if from_preds else self.predict(inputs)
        if model_first:
            for model, y_preds in models_preds.items():
                results[model] = {metric: func(targets, y_preds) 
                                  for metric, func in self.METRICS.items()}

        else:
            for metric, func in self.METRICS.items():
                results[metric] = {model: func(targets, y_preds) for model,
                                   y_preds in models_preds.items()}

        if as_df:
            results = pd.DataFrame.from_dict(results, orient="index")
        return results

    def get_model(self, model_key):
        """

        Parameters
        ----------
        model_key : the key for model in _models


        Returns
        -------
        model object corrseponding to key if key exist
        None if key does not exist

        """
        return self._models.get(model_key, None)

    def __getitem__(self, model_key):
        return self._models[model_key]

    @staticmethod
    def _is_instantiated(object_: object) -> bool:
        return not isinstance(object_, type)
