import pytest

parametrize_include_defaults = pytest.mark.parametrize('include_defaults',
                                                       ((True), (False)))