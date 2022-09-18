import pytest
from smaug import _itertools


@pytest.mark.parametrize(
    "iterable,n,expected",
    [
        pytest.param(
            [],
            10,
            [],
            id="empty iterable",
        ),
        pytest.param(
            ["A", "B", "C"],
            1,
            ["A", "B", "C"],
            id="repeat once",
        ),
        pytest.param(
            ["A", "B", "C"],
            3,
            ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            id="repeat three times",
        ),
    ],
)
def test_repeat_items(iterable, n, expected):
    output = _itertools.repeat_items(iterable, n)
    assert expected == list(output)
