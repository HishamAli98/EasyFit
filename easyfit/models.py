from abc import ABC, abstractmethod
from typing import Dict

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import (Lasso, LinearRegression, LogisticRegression,
                                  Ridge, RidgeClassifier, RidgeClassifierCV,
                                  SGDClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.base import BaseEstimator

REGRESSORS = {
    'Linear Regressor': LinearRegression,
    'Lasso Regressor': Lasso,
    'Ridge Regressor': Ridge,
    'Decision Tree Regressor': DecisionTreeRegressor,
    'Gaussian Process Regressor': GaussianProcessRegressor,
    'Support Vector Regressor': SVR,
    'XGB Regressor': XGBRegressor,
    'XGBRF Regressor': XGBRFRegressor
}

CLASSIFIERS = {
    'DecisionTreeClassifier': DecisionTreeClassifier,
    'DummyClassifier': DummyClassifier,
    'KNeighborsClassifier': KNeighborsClassifier,
    'MLPClassifier': MLPClassifier,
    'RandomForestClassifier': RandomForestClassifier,
    'RidgeClassifier': RidgeClassifier,
    'RidgeClassifierCV': RidgeClassifierCV,
    'SGDClassifier': SGDClassifier,
    'LogisticRegression': LogisticRegression,
    'SVC': SVC
}


class _EazyModel(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def score(self, X, y):
        pass


class DefaultModel(_EazyModel):
    def __init__(self, models_dict: Dict[str, BaseEstimator]):
        self._models = {model_key: model_class()
                        for model_key, model_class in models_dict.items()}

    def fit(self, X, y):
        for model in self._models.values():
            model.fit(X, y)

    def predict(self, X):
        preds = {}
        for model_key, model in self._models.items():
            preds[model_key] = model.predict(X)
        return preds

    def score(self, X, y):
        scores = {}
        for model_key, model in self._models.items():
            scores[model_key] = model.score(X, y)
        return scores

    def get_model(self, model_key):
        return self._models.get(model_key, None)
    
    def __getitem__(self, model_key):
        return self._models[model_key]