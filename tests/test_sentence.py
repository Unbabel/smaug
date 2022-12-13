import pytest

from smaug import sentence

from typing import Optional


@pytest.mark.parametrize(
    "old,new,modification",
    [
        pytest.param(
            'Sentence with "old text to be replaced" in the middle.',
            'Sentence with "replaced new text" in the middle.',
            sentence.Modification(
                '"old text to be replaced"',
                '"replaced new text"',
                14,
            ),
            id="Modify the middle of the sentence.",
        ),
        pytest.param(
            '"Sentence with old text to be replaced" in the beginning.',
            '"Sentence with replaced new text" in the beginning.',
            sentence.Modification(
                '"Sentence with old text to be replaced"',
                '"Sentence with replaced new text"',
                0,
            ),
            id="Modify the beginning of the sentence.",
        ),
        pytest.param(
            'Sentence with "old text to be replaced in the end".',
            'Sentence with "replaced new text in the end".',
            sentence.Modification(
                '"old text to be replaced in the end".',
                '"replaced new text in the end".',
                14,
            ),
            id="Modify the end of the sentence.",
        ),
    ],
)
def test_modification(old: str, new: str, modification: sentence.Modification):
    new_output = modification.apply(old)
    assert new == new_output
    old_output = modification.reverse(new_output)
    assert old == old_output


def test_modification_trace():
    old = 'Original Sentence with "text to modify" and "more text to modify".'
    new = 'Original Sentence with "modified text" and "more modifed text".'

    trace = sentence.ModificationTrace.from_modifications(
        sentence.Modification('"text to modify"', '"modified text"', 23),
        sentence.Modification('"more text to modify"', '"more modifed text"', 43),
    )

    new_output = trace.apply(old)
    assert new == new_output
    old_output = trace.reverse(new_output)
    assert old == old_output


@pytest.mark.parametrize(
    "original,span,idx,expected",
    [
        pytest.param(
            sentence.Sentence("Original Sentence without modifications."),
            ", text to add at the middle,",
            17,
            sentence.Sentence(
                "Original Sentence, text to add at the middle, without modifications.",
                trace=sentence.ModificationTrace.from_modifications(
                    sentence.Modification("", ", text to add at the middle,", 17),
                ),
            ),
            id="Middle of sentence insertion.",
        ),
        pytest.param(
            sentence.Sentence("Original Sentence without modifications."),
            "Text to add at the beginning. ",
            0,
            sentence.Sentence(
                "Text to add at the beginning. Original Sentence without modifications.",
                trace=sentence.ModificationTrace.from_modifications(
                    sentence.Modification("", "Text to add at the beginning. ", 0),
                ),
            ),
            id="Beginning of sentence insertion.",
        ),
        pytest.param(
            sentence.Sentence("Original Sentence without modifications."),
            " Text to add at the end.",
            40,
            sentence.Sentence(
                "Original Sentence without modifications. Text to add at the end.",
                trace=sentence.ModificationTrace.from_modifications(
                    sentence.Modification("", " Text to add at the end.", 40),
                ),
            ),
            id="End of sentence replacement.",
        ),
    ],
)
def test_insert(
    original: sentence.Sentence, span: str, idx: int, expected: sentence.Sentence
):
    output = original.insert(span, idx)
    _assert_equal_sentences(expected, output)


@pytest.mark.parametrize(
    "original,loc,expected",
    [
        pytest.param(
            sentence.Sentence(
                "Original Sentence, text to delete at the middle, without modifications."
            ),
            (17, 48),
            sentence.Sentence(
                "Original Sentence without modifications.",
                trace=sentence.ModificationTrace.from_modifications(
                    sentence.Modification(", text to delete at the middle,", "", 17),
                ),
            ),
            id="Middle of sentence deletion.",
        ),
        pytest.param(
            sentence.Sentence(
                "Text to delete at the beginning. Original Sentence without modifications."
            ),
            (0, 33),
            sentence.Sentence(
                "Original Sentence without modifications.",
                trace=sentence.ModificationTrace.from_modifications(
                    sentence.Modification("Text to delete at the beginning. ", "", 0),
                ),
            ),
            id="Beginning of sentence deletion.",
        ),
        pytest.param(
            sentence.Sentence(
                "Original Sentence without modifications. Text to delete at the end."
            ),
            (40, 67),
            sentence.Sentence(
                "Original Sentence without modifications.",
                trace=sentence.ModificationTrace.from_modifications(
                    sentence.Modification(" Text to delete at the end.", "", 40),
                ),
            ),
            id="End of sentence deletion.",
        ),
    ],
)
def test_deletion(
    original: sentence.Sentence,
    loc: sentence.SpanIndexLike,
    expected: sentence.Sentence,
):
    output = original.delete(loc)
    _assert_equal_sentences(expected, output)


@pytest.mark.parametrize(
    "original,span,loc,expected",
    [
        pytest.param(
            sentence.Sentence("Original Sentence without modifications."),
            ', text to replace " without",',
            (17, 25),
            sentence.Sentence(
                'Original Sentence, text to replace " without", modifications.',
                trace=sentence.ModificationTrace.from_modifications(
                    sentence.Modification(
                        " without", ', text to replace " without",', 17
                    ),
                ),
            ),
            id="Middle of sentence replacement.",
        ),
        pytest.param(
            sentence.Sentence("Original Sentence without modifications."),
            "Text to replace Original.",
            (0, 8),
            sentence.Sentence(
                "Text to replace Original. Sentence without modifications.",
                trace=sentence.ModificationTrace.from_modifications(
                    sentence.Modification("Original", "Text to replace Original.", 0),
                ),
            ),
            id="Beginning of sentence replacement.",
        ),
        pytest.param(
            sentence.Sentence("Original Sentence without modifications."),
            '. Text to replace " modifications.".',
            (25, 40),
            sentence.Sentence(
                'Original Sentence without. Text to replace " modifications.".',
                trace=sentence.ModificationTrace.from_modifications(
                    sentence.Modification(
                        " modifications.", '. Text to replace " modifications.".', 25
                    ),
                ),
            ),
            id="End of sentence replacement.",
        ),
    ],
)
def test_replace(
    original: sentence.Sentence,
    span: str,
    loc: sentence.SpanIndexLike,
    expected: sentence.Sentence,
):
    output = original.replace(span, loc)
    _assert_equal_sentences(expected, output)


def _assert_equal_sentences(expected: sentence.Sentence, actual: sentence.Sentence):
    assert expected.value == actual.value
    _assert_equal_traces(expected.trace, actual.trace)


def _assert_equal_traces(
    expected: Optional[sentence.ModificationTrace],
    actual: Optional[sentence.ModificationTrace],
):
    if expected is None:
        assert actual is None
    else:
        assert actual is not None
        assert expected.curr == actual.curr
        _assert_equal_traces(expected.prev, actual.prev)
