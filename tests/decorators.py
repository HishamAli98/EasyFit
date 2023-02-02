import pytest

parametrize_include_defaults = pytest.mark.parametrize('include_defaults',
                                                       ((True), (False)))

parametrize_score_params = pytest.mark.parametrize(
    ('as_df', 'sort'),
    (
        (True, True),
        (True, False),
        (False, True),
        (False, False)
    )
)