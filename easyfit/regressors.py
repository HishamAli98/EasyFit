import math
from typing import Dict

from sklearn.dummy import DummyRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import (BayesianRidge, ElasticNet, Lasso,
                                  LinearRegression, Ridge, SGDRegressor)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor, XGBRFRegressor

from ._models import EasyModel


class EasyRegressor(EasyModel):
    """ """
    DEFAULT_REGRESSORS = {
        'Dummy Regressor': DummyRegressor,
        'Linear Regressor': LinearRegression,
        'Lasso Regressor': Lasso,
        'Ridge Regressor': Ridge,
        'Bayesian Ridge Regressor': BayesianRidge,
        'Elastic Net Regressor': ElasticNet,
        'SGD Regressor': SGDRegressor,
        'Decision Tree Regressor': DecisionTreeRegressor,
        'Gaussian Process Regressor': GaussianProcessRegressor,
        'Support Vector Regressor': SVR,
        'Linear SVR': LinearSVR,
        'XGB Regressor': XGBRegressor,
        'XGBRF Regressor': XGBRFRegressor,
        'MLP Regressor': MLPRegressor
    }
    METRICS = {
        "Mean Absolute Error": mean_absolute_error,
        "Mean Squared Error": mean_squared_error,
        "Root Mean Squared Error": lambda X, y: math.sqrt(mean_squared_error(X, y)),
        "R2 Score": r2_score
    }

    def __init__(self, models_dict: Dict = None,
                 include_defaults: bool = True):
        super().__init__(self.DEFAULT_REGRESSORS, models_dict,
                         include_defaults=include_defaults)
