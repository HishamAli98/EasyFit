import pytest
import sklearn
from packaging import version
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier

from easyfit.classifiers import EasyClassifier
from easyfit.exceptions import InvalidModelError

if version.parse(sklearn.__version__) >= version.parse('0.22.0'):
    from sklearn.utils._testing import ignore_warnings
else:
    from sklearn.utils.testing import ignore_warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from .decorators import (parametrize_evaluate_params,
                         parametrize_include_defaults,
                         parametrize_score_params)


class TestConstructor:
    def test_no_models_supplied(self):
        """
        when models_dict = None or {} and include_defaults = False in constructor
        not passed, ValueError is raised; because there will be no models to train
        """
        with pytest.raises(ValueError) as exception_info:
            EasyClassifier(include_defaults=False)
        assert exception_info.match("must supply models_dict or set include_defaults to True")


    @parametrize_include_defaults
    def test_models_dict_wrong_type(self, include_defaults):
        """
        When type(models_dict) != dict, raise TypeError
        """
        models_list = ['models_1', 'model_2']
        with pytest.raises(TypeError) as exception_info:
            EasyClassifier(models_dict=models_list,
                                include_defaults=include_defaults)
        assert exception_info.match("models_dict must be of type dict")


    @parametrize_include_defaults
    def test_invalid_models(self, invalid_models, include_defaults):
        """
        When models_dict contains a model that does not have predict() ot fit() method,
        construcor should raise easyfit.exceptions.InvalidModelError
        """
        models_dict, missing = invalid_models
        with pytest.raises(InvalidModelError) as exception_info:
            EasyClassifier(models_dict=models_dict,
                                include_defaults=include_defaults)
        assert exception_info.match(f'model model_1 must have {missing} method')


    @parametrize_include_defaults
    def test_valid_models(self, valid_models, include_defaults):
        """
        When models_dict contains models objects and classes, model objects should be added
        to EasyClassifier._models as is, and classes should be instantiated then added
        """
        models_dict = valid_models
        model = EasyClassifier(models_dict=models_dict,
                                include_defaults=include_defaults)
        for model_key, model_obj in model._models.items():
            assert not isinstance(model_obj, type), f'Model {model_key} class not instantiated'

model = EasyClassifier()
data = load_iris(as_frame=True)
X = data.data
y = data.target

class TestFit:
    @ignore_warnings(category=ConvergenceWarning)
    @pytest.mark.filterwarnings('ignore::DeprecationWarning')
    def test_fit(self):
        model.fit(X, y)


class TestPredict:
    @ignore_warnings(category=ConvergenceWarning)
    @pytest.mark.filterwarnings('ignore::DeprecationWarning')
    @pytest.mark.filterwarnings('ignore::FutureWarning')
    def test_predict(self):

        preds = model.predict(X)

        assert isinstance(preds, dict)


class TestScore:
    @ignore_warnings(category=ConvergenceWarning)
    @pytest.mark.filterwarnings('ignore::DeprecationWarning')
    @pytest.mark.filterwarnings('ignore::FutureWarning')
    @parametrize_score_params
    def test_score(self, as_df, sort):
        result = model.score(X, y, as_df=as_df, sort=sort)

        if as_df:
            assert isinstance(result, pd.DataFrame), (
                'result type is not pd.Dataframe when as_df=True'
            )
            assert list(result.columns) == ["Model", "Score"], (
                'result.columns != ["Model", "Score"]'
            )
            if sort:
                assert result['Score'].is_monotonic_decreasing, (
                    'Scores not sorted in decreasing order when sort=True'
                )
        else:
            assert isinstance(result, dict), (
                'result type is not dict when as_df=False'
            )
            if sort:
                assert list(result.values()) == sorted(result.values(), 
                                                       reverse=True)


class TestEvaluate:
    @parametrize_evaluate_params
    def test_evaluate(self, as_df, model_first, from_preds):
        if from_preds:
            preds = model.predict(X)
            result = model.evaluate(preds, y, as_df=as_df, 
                                   model_first=model_first, 
                                   from_preds=from_preds)
        else:
            result = model.evaluate(X, y, as_df=as_df, 
                                   model_first=model_first, 
                                   from_preds=from_preds)

        if as_df:
            assert isinstance(result, pd.DataFrame), (
                'result type is not pd.Dataframe when as_df=True'
            )
            if model_first:
                assert list(result.columns) == list(model._METRICS.keys()), (
                    'result.columns != model._METRICS.keys()'
                )
            else:
                assert list(result.columns) == list(model._models.keys()), (
                    'result.columns != model._models.keys()'
                )
        else:
            assert isinstance(result, dict), (
                'result type is not dict when as_df=False'
            )
            if model_first:
                assert result.keys() == model._models.keys(), (
                    'result.keys() != model._METRICS.keys()'
                )
            else:
                assert result.keys() == model._METRICS.keys(), (
                    'result.keys() != model._models.keys()'
                )
            
class TestGetModel:
    def test_model_exist(self):
        dummy = model.get_model('Dummy Classifier')
        assert isinstance(dummy, DummyClassifier), (
        "incorrect model returned in get_model"
        )

    def test_model_does_not_exist(self):
        dummy = model.get_model('Missing Model')
        assert dummy is None, (
        "get model does not return None when model missing"
        )