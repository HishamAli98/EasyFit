import pytest


@pytest.fixture
def para():
    def func():
        return pytest.mark.parametrize('include_defaults',((True), (False)))
    return func

@pytest.fixture
def model_without_fit():
    class ModelWithoutFit:
        def predict(self):
            pass
    return ModelWithoutFit()

@pytest.fixture
def model_without_predict():
    class ModelWithoutPredict:
        def fit(self):
            pass
    return ModelWithoutPredict()

@pytest.fixture
def valid_model():
    class ValidModel:
        def fit(self):
            pass

        def predict(self):
            pass
    return ValidModel()