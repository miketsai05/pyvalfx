import numpy as np
import pytest
from pytest import param

from pyvalfx.cfi.binomial import BinomialAmerican, BinomialCRR
from pyvalfx.cfi.blackscholes import BlackScholes

dt = 1 / 252


@pytest.mark.parametrize(
    "S, K, T, sigma, r, q, error_message",
    [
        # Test normal methods and properties
        param(10, 10, 5, 0.45, 0.05, 0, None, id="test_normal_1"),
        param(30, 25, 3, 0.20, 0.01, 0, None, id="test_normal_2"),
        param(10, 15, 8, 0.90, 0.03, 0, None, id="test_normal_3"),
        param(10, 10, 5, 0.45, 0.03, 0.01, None, id="test_normal_4"),
    ],
)
def test_binomial(S, K, T, sigma, r, q, error_message):
    if error_message is None:
        binomial = BinomialCRR(S, T, sigma, r, round(T / dt), q)
        assert binomial.rollback_lattice(lambda x: x, lambda x: 0)[0, 0] == pytest.approx(S * np.exp(-q * T))
        assert binomial.rollback_lattice(lambda x: np.maximum(x - K, 0), lambda x: 0)[0, 0] == pytest.approx(
            BlackScholes(S, K, T, sigma, r, q).call_price(), rel=0.001
        )
        assert binomial.rollback_lattice(lambda x: np.maximum(K - x, 0), lambda x: 0)[0, 0] == pytest.approx(
            BlackScholes(S, K, T, sigma, r, q).put_price(), rel=0.001
        )

        binomial_american = BinomialAmerican(S, K, T, sigma, r, round(T / dt), q)

        if q == 0:
            assert binomial_american.call_price() == pytest.approx(
                BlackScholes(S, K, T, sigma, r, q).call_price(), rel=0.001
            )
        else:
            assert binomial_american.call_price() >= BlackScholes(S, K, T, sigma, r, q).call_price()

        assert binomial_american.put_price() >= BlackScholes(S, K, T, sigma, r, q).put_price()

    # else:
    #     with pytest.raises(ValueError) as e:
    #
    #     assert error_message in e.value.args[0]
