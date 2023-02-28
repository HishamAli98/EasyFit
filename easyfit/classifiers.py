from functools import partial
from typing import Dict

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import (LogisticRegression, RidgeClassifier,
                                  RidgeClassifierCV, SGDClassifier)
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from ._models import _EasyModel


class EasyClassifier(_EasyModel):
    """
    Fit classifier models in
        - _DEFAULT_CLASSIFIERS (if include_defaults=True)
        - models_dict (if models_dict != None)

    Parameters
    ----------
    models_dict : Dictionary of additional models
        Can hold:
            - classes: models_dict = {'LinearRegression': LinearRegression}
            - objects: models_dict = {'LinearRegression': LinearRegression()}
        (Default value = None)

    include_defaults : boolean
        Include _DEFAULT_CLASSIFIERS in trained models\n
        (Default value = True)
    """
    _DEFAULT_CLASSIFIERS = {
        'DummyClassifier': DummyClassifier,
        'LogisticRegression': LogisticRegression,
        'RidgeClassifier': RidgeClassifier,
        'RidgeClassifier CV': RidgeClassifierCV,
        'LinearSVC': LinearSVC,
        'SupportVectorClassifier': SVC,
        'DecisionTree Classifier': DecisionTreeClassifier,
        'RandomForestClassifier': RandomForestClassifier,
        'KNeighborsClassifier': KNeighborsClassifier,
        'MLPClassifier': MLPClassifier,
        'AdaBoostClassifier': AdaBoostClassifier,
        'SGDClassifier': SGDClassifier,
        'GaussianNaive Bayes': GaussianNB,
        'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis
    }
    _METRICS = {
        "Accuracy": accuracy_score,
        "Balanced Accuracy": balanced_accuracy_score,
        "Precision": partial(precision_score, average='weighted'),
        "Recall": partial(recall_score, average='weighted'),
        "F1 Score": partial(f1_score, average='weighted')
    }

    def __init__(self, models_dict: Dict = None,
                 include_defaults: bool = True):
        super().__init__(self._DEFAULT_CLASSIFIERS, models_dict,
                         include_defaults=include_defaults)

    def fit(self, X, y):
        """
        Fit classifiers in self._models on features X with targets y\n
        Calls method fit for each model in self._models

        Parameters
        ----------
        X : array of features

        y : array of targets

        Returns
        -------
        None
        """
        super().fit(X, y)

    def predict(self, X):
        """
        Make predictions for features in X\n
        Call predict method for each model in self._models

        Parameters
        ----------
        X : array of features


        Returns
        -------
        preds: Dict
            Dictionary with same keys in self._models and predictions for each
            model of features in X

        """
        return super().predict(X)

    def score(self, X, y, as_df=True, sort=True):
        """
        Calculate score for each model in self._models\n
        Calls score method for each model in self._models\n
        Return mean accuracy for each model on the given data and labels

        Parameters
        ----------
        X : array of features

        y : array of targets

        as_df : boolean
            - if True: return results in pd.DataFrame
            - if False: return results in dictionary
            (Default value = True)
        sort: boolean
            - if True: returns results sorted in discending order by score
            - if False: returns results in the original order of models
            (Default value = True)

        Returns
        -------
        results: Dict (as_df=False) or pd.Dataframe (as_df=True)
        """
        return super().score(X, y, as_df=as_df, sort=sort)

    def evaluate(self, X, y, as_df=True, model_first=True, from_preds=False):
        """
        Returns models results on each of the metrics in self._METRICS
        dictionary

        Parameters
        ----------
        X : array of features

        y : array of targets

        as_df : boolean
            - if True: return results in pd.DataFrame
            - if False: return results in dictionary
            (Default value = True)
        model_first : boolean
            - if True: returns models at axis=0 (rows), results at axis=1 (columns)
            - if False: returns models at axis=1 (columns), results at axis=0 (rows)
            (Default value = True)
        from_preds : boolean
            - if True: make preditions then calacuate metrics (X holds input features)
            - if False: calcualte metrics from predictions (X holds predictions)
            (Default value = False)

        Returns
        -------
        results: Dict (as_df=False) or pd.Dataframe (as_df=True)

        """
        return super().evaluate(X, y, as_df=as_df, model_first=model_first,
                                from_preds=from_preds)

    def get_model(self, model_key):
        """
        Get specific model from self._models

        Parameters
        ----------
        model_key : the key for model in self._models

        Returns
        -------
        model object corrseponding to key if key exist
        None if key does not exist

        """
        return self._models.get(model_key, None)
