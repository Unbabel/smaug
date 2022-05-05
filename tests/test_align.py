import pytest
from maug import _align


@pytest.mark.parametrize(
    "a,b,expected",
    [
        pytest.param(
            "Yesterday, John went to the movies.",
            "Yesterday, Patrick went to the theatre.",
            (
                [11, 12, 13, 14, 28, 29, 30, 31, 33],
                [11, 12, 13, 14, 15, 16, 17, 31, 32, 33, 34, 35, 36],
            ),
            id="strings with single word matches",
        ),
        pytest.param(
            "Yesterday, John went to the movies.".split(),
            "Yesterday, Patrick went to the theatre.".split(),
            (
                [1, 5],
                [1, 5],
            ),
            id="lists with single word matches",
        ),
        pytest.param(
            "Today, John went to the beach.",
            "Today, Patrick went to town.",
            (
                [7, 8, 9, 10, 21, 22, 23, 24, 25, 26, 27, 28],
                [7, 8, 9, 10, 11, 12, 13, 24, 25, 26],
            ),
            id="strings with multi word matches",
        ),
        pytest.param(
            "Today, John went to the beach.".split(),
            "Today, Patrick went to town.".split(),
            (
                [1, 4, 5],
                [1, 4],
            ),
            id="lists with multi word matches",
        ),
    ],
)
def test_mismatched_indexes(a, b, expected):
    output = _align.mismatched_indexes(a, b)
    assert expected == output


@pytest.mark.parametrize(
    "a,b,expected",
    [
        pytest.param(
            "Yesterday, John went to the movies.",
            "Yesterday, Patrick went to the theatre.",
            (
                [slice(11, 15), slice(28, 32), slice(33, 34)],
                [slice(11, 18), slice(31, 37)],
            ),
            id="strings with single word matches",
        ),
        pytest.param(
            "Yesterday, John went to the movies.".split(),
            "Yesterday, Patrick went to the theatre.".split(),
            (
                [slice(1, 2), slice(5, 6)],
                [slice(1, 2), slice(5, 6)],
            ),
            id="lists with single word matches",
        ),
        pytest.param(
            "Today, John went to the beach.",
            "Today, Patrick went to town.",
            (
                [slice(7, 11), slice(21, 29)],
                [slice(7, 14), slice(24, 27)],
            ),
            id="strings with multi word matches",
        ),
        pytest.param(
            "Today, John went to the beach.".split(),
            "Today, Patrick went to town.".split(),
            (
                [slice(1, 2), slice(4, 6)],
                [slice(1, 2), slice(4, 5)],
            ),
            id="lists with multi word matches",
        ),
    ],
)
def test_mismatched_slices(a, b, expected):
    output = _align.mismatched_slices(a, b)
    assert expected == output
