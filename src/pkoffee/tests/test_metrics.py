import pytest
import numpy as np


def test_size_mismatch_valid():
    from pkoffee.metrics import check_size_match

    a = np.ones(shape=3)
    b = np.zeros(shape=3)

    check_size_match(a, b)


def test_size_mismatch_invalid():
    from pkoffee.metrics import check_size_match, SizeMismatchError

    a = np.ones(shape=3)
    b = np.zeros(shape=4)

    with pytest.raises(SizeMismatchError, match="Arrays must have same length"):
        check_size_match(a, b)


def test_r2():
    from pkoffee.metrics import compute_r2

    rng = np.random.default_rng(seed=0)
    y_true = rng.normal(size=10)

    # perfect prediction is 1.0
    assert compute_r2(y_true, y_true) == 1.0

    # predictiong mean should be 0.0
    y_pred = np.full(len(y_true), np.mean(y_true))
    assert compute_r2(y_true, y_pred) == 0.0

    # NaN returning
    y_true = np.ones(shape=10)
    assert np.isnan(compute_r2(y_true, y_pred))

def test_rmse():
    from pkoffee.metrics import compute_rmse

    rng = np.random.default_rng(seed=0)
    y_true = rng.normal(size=10)

    # perfect prediction is 0.0
    assert compute_rmse(y_true, y_true) == 0.0

    
def test_mae():
    from pkoffee.metrics import compute_rmse

    rng = np.random.default_rng(seed=0)
    y_true = rng.normal(size=10)

    # perfect prediction is 0.0
    assert compute_rmse(y_true, y_true) == 0.0