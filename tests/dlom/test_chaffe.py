import pytest
from pytest import param

from pyvalfx.dlom.chaffe import Chaffe


@pytest.mark.parametrize(
    "T, sigma, r, q, expected, error_message",
    [
        # Test normal methods and properties
        param(5, 0.45, 0.05, 0.00, 0.24152077, None, id="test_normal_1"),
        param(2, 0.20, 0.01, 0.00, 0.10172519, None, id="test_normal_2"),
        param(10, 0.90, 0.03, 0.00, 0.60811511, None, id="test_normal_3"),
        param(5, 0.45, 0.03, 0.01, 0.30522254, None, id="test_normal_4"),
        param(8, 0.30, 0.03, 0.05, 0.30153414, None, id="test_normal_5"),
    ],
)
def test_chaffe(T, sigma, r, q, expected, error_message):
    if error_message is None:
        assert Chaffe(T, sigma, r, q).calculate_dlom() == pytest.approx(expected)

    else:
        with pytest.raises(ValueError) as e:
            Chaffe(T, sigma, r, q)
        assert error_message in e.value.args[0]
