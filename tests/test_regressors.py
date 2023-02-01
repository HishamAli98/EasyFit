import pytest

from easyfit.exceptions import InvalidModelError
from easyfit.regressors import EasyRegressor

from .decorators import parametrize_include_defaults


def test_no_models_supplied():
    """
    when models_dict = None or {} and include_defaults = False in constructor
    not passed, ValueError is raised; because there will be no models to train
    """
    with pytest.raises(ValueError) as exception_info:
        model = EasyRegressor(include_defaults=False)
    assert exception_info.match("must supply models_dict or set include_defaults to True")

@parametrize_include_defaults
def test_models_dict_wrong_type(include_defaults):
    """
    When type(models_dict) != dict, raise TypeError
    """
    models_list = ['models_1', 'model_2']
    with pytest.raises(TypeError) as exception_info:
        model = EasyRegressor(models_dict=models_list,
                              include_defaults=include_defaults)
    assert exception_info.match("models_dict must be of type dict")

@parametrize_include_defaults
def test_model_without_fit(model_without_fit, valid_model, include_defaults):
    """
    When models_dict contains a model that does not have fit() method,
    construcor should raise easyfit.exceptions.InvalidModelError
    """
    model_1 = model_without_fit
    model_2 = valid_model
    models_dict={'model_1': model_1, 'model_2': model_2}
    with pytest.raises(InvalidModelError) as exception_info:
        model = EasyRegressor(models_dict=models_dict,
                              include_defaults=include_defaults)
    assert exception_info.match('model model_1 must have fit method')

@parametrize_include_defaults
def test_model_without_predict(model_without_predict, valid_model, include_defaults):
    """
    When models_dict contains a model that does not have predict() method,
    construcor should raise easyfit.exceptions.InvalidModelError
    """
    model_1 = model_without_predict
    model_2 = valid_model
    models_dict={'model_1': model_1, 'model_2': model_2}
    with pytest.raises(InvalidModelError) as exception_info:
        model = EasyRegressor(models_dict=models_dict,
                              include_defaults=include_defaults)
    assert exception_info.match('model model_1 must have predict method')
