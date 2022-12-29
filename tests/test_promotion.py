import pytest

from smaug.core import Data
from smaug.promote import promote_to_data


@pytest.mark.parametrize(
    "value,expected",
    [
        pytest.param(
            Data([1, 2, 3]),
            Data([1, 2, 3]),
            id="Data of ints",
        ),
        pytest.param(
            Data([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            Data([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            id="Data of lists of ints",
        ),
        pytest.param([1, 2, 3], Data([1, 2, 3]), id="List of ints"),
        pytest.param(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            Data([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            id="List of lists of ints",
        ),
        pytest.param(
            1,
            Data([1]),
            id="Int",
        ),
    ],
)
def test_promote_to_data(value, expected):
    promoted = promote_to_data(value)
    assert isinstance(promoted, Data)
    assert len(expected) == len(promoted)
    for e, p in zip(expected, promoted):
        assert e == p

