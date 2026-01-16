import pytest
import numpy as np

def test_Quadratic():
    from pkoffee.parametric_function import Quadratic
    
    rng = np.random.default_rng(seed=0)
    [x, a0, a1, a2] = rng.normal(size=4)
    func = Quadratic()

    np.testing.assert_equal(func(x, a0, a1, a2), a0 + a1 * x + a2 * x**2)

def test_MichaelisMentenSaturation():
    from pkoffee.parametric_function import MichaelisMentenSaturation, ParametersBounds
    from pkoffee.data import data_dtype, neg_inf, pos_inf

    rng = np.random.default_rng(seed=0)
    [x, v_max, k, y0, x_min, x_max, y_min, y_max] = rng.normal(size=8)
    func = MichaelisMentenSaturation()

    np.testing.assert_equal(func(x, v_max, k, y0), y0 + v_max * (x / np.maximum(k + x, 1e-9)))

    dict = {
        "v_max": max(data_dtype(1e-8), y_max - y_min),
        "k": max(data_dtype(1.0), 0.2 * (x_min + x_max)),
        "y0": y_min,
    }
    assert func.param_guess(x_min, x_max, y_min, y_max) == dict

    bounds = ParametersBounds(
            min={"v_max": neg_inf, "k": data_dtype(0.0), "y0": neg_inf},
            max=dict.fromkeys(["v_max", "k", "y0"], pos_inf),
        )
    assert func.param_bounds() == bounds


def test_Logistic():
    from pkoffee.parametric_function import Logistic
    
    rng = np.random.default_rng(seed=0)
    [x, L, k, x0, y0] = rng.normal(size=5)
    func = Logistic()

    np.testing.assert_equal(func(x, L, k, x0, y0), y0 + L / (1.0 + np.exp(-k * (x - x0))))

def test_PeakModel():
    from pkoffee.parametric_function import PeakModel
    
    rng = np.random.default_rng(seed=0)
    [x, a, b] = rng.normal(size=3)
    func = PeakModel()

    np.testing.assert_equal(func(x, a, b), a * x * np.exp(-x / np.maximum(b, 1e-9)))

def test_Peak2Model():
    from pkoffee.parametric_function import Peak2Model
    
    rng = np.random.default_rng(seed=0)
    [x, a, b] = rng.normal(size=3)
    func = Peak2Model()

    np.testing.assert_equal(func(x, a, b), a * (x**2) * np.exp(-x / np.maximum(b, 1e-9)))