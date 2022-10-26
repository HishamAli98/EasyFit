from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import (LogisticRegression,
                                  RidgeClassifier, RidgeClassifierCV,
                                  SGDClassifier)
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ._models import EasyModel, ModelsDict


class EasyClassifier(EasyModel):
    DEFAULT_CLASSIFIERS = {
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
    METRICS = {
        "Accuracy": accuracy_score,
        "Balanced Accuracy": balanced_accuracy_score,
        "Precision": precision_score,
        "Recall": recall_score,
        "F1 Score": f1_score
    }

    def __init__(self, models_dict: ModelsDict = None,
                 include_defaults: bool = True):
        super().__init__(self.DEFAULT_CLASSIFIERS, models_dict,
                         include_defaults=include_defaults)
