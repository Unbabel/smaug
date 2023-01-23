import pytest

from smaug import ops
from smaug.core import Sentence


@pytest.mark.parametrize(
    "original,modified,expected",
    [
        (
            Sentence("Some sentence without any special characters."),
            Sentence("Other sentence without chars."),
            0,
        ),
        (
            Sentence("Some sentence without any special characters."),
            Sentence("Other sentence with <special chars>."),
            2,
        ),
        (
            Sentence("Some sentence with some <special characters."),
            Sentence("Other sentence with mode <special chars]."),
            1,
        ),
        (
            Sentence("Sentence with two special characters[]."),
            Sentence("Other sentence with many <<<special chars]."),
            3,
        ),
    ],
)
def test_character_insertions(original: Sentence, modified: Sentence, expected: int):
    output = ops.character_insertions(original, modified, chars="<>()[]{}")
    assert expected == output


@pytest.mark.parametrize(
    "s1,s2,expected",
    [
        (
            Sentence("Original sentence with 123 and .234 that should be kept."),
            Sentence("Critical sentence with .3124 and 23,234 to keep."),
            True,
        ),
        (
            Sentence("Original sentence with 12,23.42 and 12334 to keep."),
            Sentence("Critical sentence with 42 and 1.23.41 to keep."),
            True,
        ),
        (
            Sentence("Original sentence with 123 and .234 that not to keep."),
            Sentence("Critical sentence with only .3124 not to keep."),
            False,
        ),
        (
            Sentence("Original sentence with 1.232,00 and 12.23 not to keep."),
            Sentence("Critical sentence with only 1.232,00 not to keep."),
            False,
        ),
    ],
)
def test_equal_number_count(s1: Sentence, s2: Sentence, expected: bool):
    output = ops.equal_numbers_count(s1, s2)
    assert expected == output
