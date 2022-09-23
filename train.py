import pandas as pd
import sklearn as sk
import yaml
from sklearn.datasets import load_diabetes, load_iris

from easy.models import CLASSIFIERS, REGRESSORS

# load config
with open('config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
models = config['models']
task = config['task']


# load dataset
if task == 'regression':
    diabetes_dataset = load_diabetes(as_frame=True)
    diabetes_df = diabetes_dataset['frame']
    X = diabetes_df.drop('target', axis=1)
    y = diabetes_df['target']
    all_models = REGRESSORS
elif task == 'classification':
    iris = load_iris(as_frame=True)
    iris = iris['frame']
    X = iris.drop('target', axis=1)
    y = iris['target']
    all_models = CLASSIFIERS

scores = {}

for m in models:
    model = all_models[m]()
    model.fit(X, y)
    scores[m] = model.score(X, y)
    # mses[m] = sk.metrics.mean_squared_error(model.predict(X), y)

scores = pd.DataFrame(pd.Series(scores)).reset_index()
scores.columns = ['Model', 'Score']
# scores['MSE'] = mses.values()


scores.to_excel('results/result.xlsx', index=False)
print(scores)
