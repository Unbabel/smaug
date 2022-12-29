import pytest

from smaug.broadcast import broadcast_data
from smaug.core import Data

@pytest.mark.parametrize(
    "values,expected",
    [
        pytest.param(
            (Data([1]), Data([2]), Data([3])),
            (Data([1]), Data([2]), Data([3])),
            id="All with length 1",
        ),
        pytest.param(
            (Data([1, 2, 3]), Data([2, 3, 4]), Data([3, 4, 5])),
            (Data([1, 2, 3]), Data([2, 3, 4]), Data([3, 4, 5])),
            id="All with length 3",
        ),
        pytest.param(
            (Data([1, 2, 3]), Data([2]), Data([3])),
            (Data([1, 2, 3]), Data([2, 2, 2]), Data([3, 3, 3])),
            id="One with length 3, two with length 1",
        ),
        pytest.param(
            (Data([1]), Data([2, 3, 4]), Data([3, 4, 5])),
            (Data([1, 1, 1]), Data([2, 3, 4]), Data([3, 4, 5])),
            id="Twi with length 3, one with length 1",
        ),
    ],
)
def test_broadcast_data(values, expected):
    broadcasted = broadcast_data(*values)
    for expected_value, promoted_value in zip(expected, broadcasted):
        assert isinstance(promoted_value, Data)
        assert len(expected_value) == len(promoted_value)
        for e, p in zip(expected_value, promoted_value):
            assert e == p
