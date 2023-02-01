import pytest
from easyfit._models import _EasyModel


def test_create_easy_model():
    '''
    Object creation form abstract class easyfit._model._EasyModel
    should raise NotImplementedError
    '''
    with pytest.raises(NotImplementedError):
        model = _EasyModel()

