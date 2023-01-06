import pytest

from smaug import ops
from smaug.core import Data
from smaug.frozen import frozenlist

from typing import Tuple


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "Test with 1.23,22 number and another 1.220.",
            Data([frozenlist([(10, 17), (37, 42)])]),
        ),
        (
            "Test with .21234 number and another 2312234.",
            Data([frozenlist([(10, 16), (36, 43)])]),
        ),
    ],
)
def test_detect_numbers(text: str, expected: Data[frozenlist[Tuple[int, int]]]):
    output = ops.regex_detect_numbers(text)
    assert isinstance(output, Data)
    assert len(expected) == len(output)
    for e, o in zip(expected, output):
        assert e == o
