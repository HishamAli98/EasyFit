import pytest


class ModelWithoutPredict:
    def fit(self):
        pass


class ModelWithoutFit:
    def predict(self):
        pass


class ValidModel:
    def fit(self):
        pass

    def predict(self):
        pass


@pytest.fixture(params=[
    ({'model_1': ModelWithoutFit(), 'model_2': ValidModel()}, 'fit'),
    ({'model_1': ModelWithoutPredict(), 'model_2': ValidModel()}, 'predict'),
    ({'model_1': ModelWithoutFit, 'model_2': ValidModel}, 'fit'),
    ({'model_1': ModelWithoutPredict, 'model_2': ValidModel}, 'predict')
])
def invalid_models(request):
    return request.param


@pytest.fixture(params=[
    {'model_1': ValidModel(), 'model_2': ValidModel()},
    {'model_1': ValidModel, 'model_2': ValidModel()},
    {'model_1': ValidModel(), 'model_2': ValidModel},
    {'model_1': ValidModel, 'model_2': ValidModel}
])
def valid_models(request):
    return request.param
