import pytest

parametrize_include_defaults = pytest.mark.parametrize('include_defaults',
                                                       ((True), (False)))

parametrize_score_params = pytest.mark.parametrize(
    ('as_df', 'sort'),
    (
        (True,  True),
        (True,  False),
        (False, True),
        (False, False)
    )
)


parametrize_evaluate_params = pytest.mark.parametrize(
    ('as_df', 'model_first', 'from_preds'),
    (
        (True,  True,  True),
        (True,  True,  False),
        (True,  False, True),
        (True,  False, False),
        (False, True,  True),
        (False, True,  False),
        (False, False, True),
        (False, False, False)
    )
)