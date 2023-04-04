import pandas as pd
import sklearn as sk
import yaml
from sklearn.datasets import load_diabetes, load_iris

from easyfit.regressors import EasyRegressor


diabetes_dataset = load_diabetes(as_frame=True)
diabetes_df = diabetes_dataset['frame']
X = diabetes_df.drop('target', axis=1)
y = diabetes_df['target']

model = sk.linear_model.LinearRegression()

reg = EasyRegressor({'model': model}, )
reg.fit(X, y)
print(reg.score(X, y))
