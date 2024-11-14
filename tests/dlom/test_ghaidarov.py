import pytest
from pytest import param

from pyvallib.dlom.ghaidarov import Ghaidarov


@pytest.mark.parametrize(
    "T, sigma, q, expected, error_message",
    [
        # Test normal methods and properties
        param(5, 0.45, 0.00, 0.23808668, None, id="test_normal_1"),
        param(2, 0.20, 0.00, 0.06529152, None, id="test_normal_2"),
        param(10, 0.90, 0.00, 0.71679876, None, id="test_normal_3"),
        param(5, 0.45, 0.01, 0.22647505, None, id="test_normal_4"),
        param(8, 0.30, 0.05, 0.13357704, None, id="test_normal_5"),
    ],
)
def test_ghaidarov(T, sigma, q, expected, error_message):
    if error_message is None:
        assert Ghaidarov(T, sigma, q).calculate_dlom() == pytest.approx(expected)

    else:
        with pytest.raises(ValueError) as e:
            Ghaidarov(T, sigma, q)
        assert error_message in e.value.args[0]
