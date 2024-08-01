import numpy as np
import pytest
from pytest import param

from pyvalfx.cfi.blackscholes import BlackScholes

err_msg_S_K = "Expected inputs S, K to be greater than or equal to 0"
err_msg_T_sigma_r = "Expected inputs T, sigma, rfr to be greater than 0"
err_msg_array_dim = "All non-scalar inputs must have the same dimensions"


@pytest.mark.parametrize(
    "S, K, T, sigma, r, q, expected, error_message",
    [
        # Test normal methods and properties
        param(10, 10, 5, 0.45, 0.05, 0, [4.62719983, 2.41520766], None, id="test_normal_1"),
        param(30, 25, 3, 0.20, 0.01, 0, [7.26986603, 1.53100437], None, id="test_normal_2"),
        param(10, 15, 8, 0.90, 0.03, 0, [7.79672799, 9.59614590], None, id="test_normal_3"),
        param(10, 10, 5, 0.45, 0.03, 0.01, [3.95743983, 3.05222535], None, id="test_normal_4"),
        # Test with stock price at $0
        param(0, 10, 5, 0.45, 0.05, 0, [0, 10 * np.exp(-5 * 0.05)], None, id="test_w_$0_S"),
        # Test with strike price at $0
        param(10, 0, 5, 0.45, 0.05, 0, [10, 0], None, id="test_w_$0_K"),
        # Test with T at ~0
        param(30, 25, 1e-6, 0.20, 0.01, 0, [5, 0], None, id="test_w_~0_T"),
        # Test input negative stock price
        param(-10, 10, 5, 0.45, 0.05, 0, [0, 0], err_msg_S_K, id="test_<$0_S"),
        # Test input negative strike price
        param(10, -10, 5, 0.45, 0.05, 0, [0, 0], err_msg_S_K, id="test_<$0_K"),
        # Test input 0 volatility
        param(10, 10, 5, 0.00, 0.05, 0, [0, 0], err_msg_T_sigma_r, id="test_w_0_sigma"),
        # Test input negative volatility
        param(10, 10, 5, -0.45, 0.05, 0, [0, 0], err_msg_T_sigma_r, id="test_w_<0_sigma"),
        # Test input 0 risk-free rate
        param(10, 10, 5, 0.45, 0.00, 0, [0, 0], err_msg_T_sigma_r, id="test_w_0_r"),
        # Test input negative risk-free rate
        param(10, 10, 5, 0.45, -0.05, 0, [0, 0], err_msg_T_sigma_r, id="test_w_<0_r"),
        # Test input 0 T
        param(10, 10, 0, 0.45, 0.05, 0, [0, 0], err_msg_T_sigma_r, id="test_w_0_T"),
        # Test input negative T
        param(10, 10, -5, 0.45, 0.05, 0, [0, 0], err_msg_T_sigma_r, id="test_w_<0_T"),
    ],
)
def test_blackscholes(S, K, T, sigma, r, q, expected, error_message):
    if error_message is None:
        assert BlackScholes(S, K, T, sigma, r, q).call_price() == pytest.approx(expected[0])
        assert BlackScholes(S, K, T, sigma, r, q).put_price() == pytest.approx(expected[1])

    else:
        with pytest.raises(ValueError) as e:
            BlackScholes(S, K, T, sigma, r, q)
        assert error_message in e.value.args[0]


@pytest.mark.parametrize(
    "S, K, T, sigma, r, q, expected, error_message",
    [
        # Test normal methods and properties
        param([10, 15], 10, 5, 0.45, 0.05, 0, [4.62719983, 8.78479616], None, id="test_array_S"),
        param(10, [10, 15], 5, 0.45, 0.05, 0, [4.62719983, 3.37960897], None, id="test_array_K"),
        param(10, 10, [5, 10], 0.45, 0.05, 0, [4.62719983, 6.38113271], None, id="test_array_T"),
        param(10, 10, 5, [0.45, 0.60], 0.05, 0, [4.62719983, 5.59760916], None, id="test_array_sigma"),
        param(10, 10, 5, 0.45, [0.05, 0.03], 0, [4.62719983, 4.31582013], None, id="test_array_r"),
        # Test multiple arrays
        param([10, 16], [10, 12], 5, 0.45, 0.05, 0, [4.62719983, 8.81460756], None, id="test_array_S_K"),
        # Test array with different dimensions
        param([10, 16], [10, 12, 15], 5, 0.45, 0.05, 0, [], err_msg_array_dim, id="test_array_S_K"),
    ],
)
def test_blackscholes_array(S, K, T, sigma, r, q, expected, error_message):
    if error_message is None:
        assert BlackScholes(S, K, T, sigma, r, q).call_price() == pytest.approx(expected)

    else:
        with pytest.raises(ValueError) as e:
            BlackScholes(S, K, T, sigma, r, q)
        assert error_message in e.value.args[0]
