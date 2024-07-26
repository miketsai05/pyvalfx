import numpy as np
import pytest
from pytest import param

from pyvalfx.cfi.blackscholes import BlackScholes
from pyvalfx.cfi.montecarlo import MonteCarlo

err_msg_T = "Expected input T to be array or scalar"
err_msg_S_T_sigma_r = "Expected inputs S, T, sigma, rfr to be greater than 0"
n = 1e6


def numerical_call(MC: MonteCarlo, K: float):
    """
    Simple function to test Monte Carlo simulation call price is consistent with Black Scholes
    """
    payoff_fv = np.maximum(MC.generate_paths() - K, 0)
    payoff_pv = payoff_fv * np.exp(-MC.r * MC.T)
    call_price = np.mean(payoff_pv)
    return {"call_price": call_price, "standard_error": np.std(payoff_pv) / np.sqrt(MC.n)}


@pytest.mark.parametrize(
    "S, K, T, sigma, r, q, error_message",
    [
        # Test normal methods and properties
        param(10, 10, 5, 0.45, 0.05, 0, None, id="test_normal_1"),
        param(30, 25, 3, 0.20, 0.01, 0, None, id="test_normal_2"),
        param(10, 15, 8, 0.90, 0.03, 0, None, id="test_normal_3"),
        # Test string in T
        param(10, 10, "test", 0.45, 0.05, 0, err_msg_T, id="test_w_str_T"),
        # Test input 0 volatility
        param(10, 10, 5, 0.00, 0.05, 0, err_msg_S_T_sigma_r, id="test_w_0_sigma"),
        # Test input negative volatility
        param(10, 10, 5, -0.45, 0.05, 0, err_msg_S_T_sigma_r, id="test_w_<0_sigma"),
        # Test input 0 T
        param(10, 10, 0, 0.45, 0.05, 0, err_msg_S_T_sigma_r, id="test_w_0_T"),
        # Test input negative T
        param(10, 10, -5, 0.45, 0.05, 0, err_msg_S_T_sigma_r, id="test_w_<0_T"),
    ],
)
def test_montecarlo(S, K, T, sigma, r, q, error_message):
    if error_message is None:
        mc_call = numerical_call(MonteCarlo(S, T, sigma, r, n, q), K)

        assert mc_call["call_price"] == pytest.approx(
            BlackScholes(S, K, T, sigma, r, q).call_price(), abs=1.96 * mc_call["standard_error"]
        )

    else:
        with pytest.raises(ValueError) as e:
            MonteCarlo(S, T, sigma, r, n)
        assert error_message in e.value.args[0]