import pytest
from pytest import param

from pyvallib.dlom.differential_put import DifferentialPut


@pytest.mark.parametrize(
    "T, sigma_preferred, sigma_common, r, q, expected, error_message",
    [
        # Test normal methods and properties
        param(5, 0.45, 0.60, 0.05, 0.00, 0.12794145, None, id="test_normal_1"),
        param(2, 0.20, 0.40, 0.01, 0.00, 0.12135038, None, id="test_normal_2"),
        param(10, 0.90, 1.20, 0.03, 0.00, 0.21201439, None, id="test_normal_3"),
        param(5, 0.45, 0.90, 0.03, 0.01, 0.39044997, None, id="test_normal_4"),
        param(8, 0.30, 0.60, 0.03, 0.05, 0.28353183, None, id="test_normal_5"),
    ],
)
def test_differential_put(T, sigma_preferred, sigma_common, r, q, expected, error_message):
    if error_message is None:
        assert DifferentialPut(T, sigma_preferred, sigma_common, r, q).calculate_dlom() == pytest.approx(expected)

    else:
        with pytest.raises(ValueError) as e:
            DifferentialPut(T, sigma_preferred, sigma_common, r, q)
        assert error_message in e.value.args[0]
