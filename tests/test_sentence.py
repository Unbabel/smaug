import pytest

from smaug import ops
from smaug.core import Sentence, Modification, ModificationTrace, SpanIndexLike

from typing import Optional


@pytest.mark.parametrize(
    "original,span,idx,expected",
    [
        pytest.param(
            Sentence("Original Sentence without modifications."),
            ", text to add at the middle,",
            17,
            Sentence(
                "Original Sentence, text to add at the middle, without modifications.",
                trace=ModificationTrace.from_modifications(
                    Modification("", ", text to add at the middle,", 17),
                ),
            ),
            id="Middle of sentence insertion.",
        ),
        pytest.param(
            Sentence("Original Sentence without modifications."),
            "Text to add at the beginning. ",
            0,
            Sentence(
                "Text to add at the beginning. Original Sentence without modifications.",
                trace=ModificationTrace.from_modifications(
                    Modification("", "Text to add at the beginning. ", 0),
                ),
            ),
            id="Beginning of sentence insertion.",
        ),
        pytest.param(
            Sentence("Original Sentence without modifications."),
            " Text to add at the end.",
            40,
            Sentence(
                "Original Sentence without modifications. Text to add at the end.",
                trace=ModificationTrace.from_modifications(
                    Modification("", " Text to add at the end.", 40),
                ),
            ),
            id="End of sentence replacement.",
        ),
    ],
)
def test_insert(original: Sentence, span: str, idx: int, expected: Sentence):
    output = ops.insert(original, span, idx)
    _assert_equal_sentences(expected, output)


@pytest.mark.parametrize(
    "original,loc,expected",
    [
        pytest.param(
            Sentence(
                "Original Sentence, text to delete at the middle, without modifications."
            ),
            (17, 48),
            Sentence(
                "Original Sentence without modifications.",
                trace=ModificationTrace.from_modifications(
                    Modification(", text to delete at the middle,", "", 17),
                ),
            ),
            id="Middle of sentence deletion.",
        ),
        pytest.param(
            Sentence(
                "Text to delete at the beginning. Original Sentence without modifications."
            ),
            (0, 33),
            Sentence(
                "Original Sentence without modifications.",
                trace=ModificationTrace.from_modifications(
                    Modification("Text to delete at the beginning. ", "", 0),
                ),
            ),
            id="Beginning of sentence deletion.",
        ),
        pytest.param(
            Sentence(
                "Original Sentence without modifications. Text to delete at the end."
            ),
            (40, 67),
            Sentence(
                "Original Sentence without modifications.",
                trace=ModificationTrace.from_modifications(
                    Modification(" Text to delete at the end.", "", 40),
                ),
            ),
            id="End of sentence deletion.",
        ),
    ],
)
def test_deletion(original: Sentence, loc: SpanIndexLike, expected: Sentence):
    output = ops.delete(original, loc)
    _assert_equal_sentences(expected, output)


@pytest.mark.parametrize(
    "original,span,loc,expected",
    [
        pytest.param(
            Sentence("Original Sentence without modifications."),
            ', text to replace " without",',
            (17, 25),
            Sentence(
                'Original Sentence, text to replace " without", modifications.',
                trace=ModificationTrace.from_modifications(
                    Modification(" without", ', text to replace " without",', 17),
                ),
            ),
            id="Middle of sentence replacement.",
        ),
        pytest.param(
            Sentence("Original Sentence without modifications."),
            "Text to replace Original.",
            (0, 8),
            Sentence(
                "Text to replace Original. Sentence without modifications.",
                trace=ModificationTrace.from_modifications(
                    Modification("Original", "Text to replace Original.", 0),
                ),
            ),
            id="Beginning of sentence replacement.",
        ),
        pytest.param(
            Sentence("Original Sentence without modifications."),
            '. Text to replace " modifications.".',
            (25, 40),
            Sentence(
                'Original Sentence without. Text to replace " modifications.".',
                trace=ModificationTrace.from_modifications(
                    Modification(
                        " modifications.", '. Text to replace " modifications.".', 25
                    ),
                ),
            ),
            id="End of sentence replacement.",
        ),
    ],
)
def test_replace(original: Sentence, span: str, loc: SpanIndexLike, expected: Sentence):
    output = ops.replace(original, span, loc)
    _assert_equal_sentences(expected, output)


def _assert_equal_sentences(expected: Sentence, actual: Sentence):
    assert expected.value == actual.value
    _assert_equal_traces(expected.trace, actual.trace)


def _assert_equal_traces(
    expected: Optional[ModificationTrace],
    actual: Optional[ModificationTrace],
):
    if expected is None:
        assert actual is None
    else:
        assert actual is not None
        assert expected.curr == actual.curr
        _assert_equal_traces(expected.prev, actual.prev)
