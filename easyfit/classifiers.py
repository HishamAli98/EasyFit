from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import (LogisticRegression,
                                  RidgeClassifier, RidgeClassifierCV,
                                  SGDClassifier, )
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from ._models import EasyModel, ModelsDict


class EasyClassifier(EasyModel):
    """ """
    DEFAULT_CLASSIFIERS = {
        'Dummy Classifier': DummyClassifier,
        'LogisticRegression': LogisticRegression,
        'RidgeClassifier': RidgeClassifier,
        'RidgeClassifierCV': RidgeClassifierCV,
        'LinearSVC': LinearSVC,
        'Support Vector Classifier': SVC,
        'Decision Tree Classifier': DecisionTreeClassifier,
        'RandomForestClassifier': RandomForestClassifier,
        'KNeighborsClassifier': KNeighborsClassifier,
        'MLP Classifier': MLPClassifier,
        'AdaBoost Classifier': AdaBoostClassifier,
        'SGDClassifier': SGDClassifier,
        'Gaussian Naïve Bayes': GaussianNB,
        'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis
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
