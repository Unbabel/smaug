import pytest

from smaug import core


@pytest.mark.parametrize(
    "value,expected",
    [
        pytest.param(
            core.Data([1, 2, 3]),
            core.Data([1, 2, 3]),
            id="Data of ints",
        ),
        pytest.param(
            core.Data([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            core.Data([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            id="Data of lists of ints",
        ),
        pytest.param([1, 2, 3], core.Data([1, 2, 3]), id="List of ints"),
        pytest.param(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            core.Data([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            id="List of lists of ints",
        ),
        pytest.param(
            1,
            core.Data([1]),
            id="Int",
        ),
    ],
)
def test_promote_to_data(value, expected):
    promoted = core.promote_to_data(value)
    assert isinstance(promoted, core.Data)
    assert len(expected) == len(promoted)
    for e, p in zip(expected, promoted):
        assert e == p


@pytest.mark.parametrize(
    "values,expected",
    [
        pytest.param(
            (core.Data([1]), core.Data([2]), core.Data([3])),
            (core.Data([1]), core.Data([2]), core.Data([3])),
            id="All with length 1",
        ),
        pytest.param(
            (core.Data([1, 2, 3]), core.Data([2, 3, 4]), core.Data([3, 4, 5])),
            (core.Data([1, 2, 3]), core.Data([2, 3, 4]), core.Data([3, 4, 5])),
            id="All with length 3",
        ),
        pytest.param(
            (core.Data([1, 2, 3]), core.Data([2]), core.Data([3])),
            (core.Data([1, 2, 3]), core.Data([2, 2, 2]), core.Data([3, 3, 3])),
            id="One with length 3, two with length 1",
        ),
        pytest.param(
            (core.Data([1]), core.Data([2, 3, 4]), core.Data([3, 4, 5])),
            (core.Data([1, 1, 1]), core.Data([2, 3, 4]), core.Data([3, 4, 5])),
            id="Twi with length 3, one with length 1",
        ),
    ],
)
def test_broadcast_data(values, expected):
    broadcasted = core.broadcast_data(*values)
    for expected_value, promoted_value in zip(expected, broadcasted):
        assert isinstance(promoted_value, core.Data)
        assert len(expected_value) == len(promoted_value)
        for e, p in zip(expected_value, promoted_value):
            assert e == p
