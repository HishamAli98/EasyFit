# import math

# import pandas as pd
# from sklearn.datasets import load_diabetes
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.linear_model import Lasso, LinearRegression, Ridge
# from sklearn.metrics import mean_squared_error
# from sklearn.tree import DecisionTreeRegressor

from .models import DefaultModel, REGRESSORS


class DefaultRegressor(DefaultModel):

    def __init__(self):
        super().__init__(REGRESSORS)
