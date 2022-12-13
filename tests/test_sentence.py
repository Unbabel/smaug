import pytest

from smaug import sentence

@pytest.mark.parametrize(
    "original,span,idx,expected",
    [
        pytest.param(
            sentence.Sentence("Original Sentence without modifications."),
            ", text to add at the middle,",
            17,
            sentence.Sentence(
                "Original Sentence, text to add at the middle, without modifications.", 
                modification=sentence.Modification(", text to add at the middle,", sentence.SpanIndex(17, 17)),
                parent=sentence.Sentence("Original Sentence without modifications."),
            ),
            id="Middle of sentence insertion."
        ),
        pytest.param(
            sentence.Sentence("Original Sentence without modifications."),
            "Text to add at the beginning. ",
            0,
            sentence.Sentence(
                "Text to add at the beginning. Original Sentence without modifications.", 
                modification=sentence.Modification("Text to add at the beginning. ", sentence.SpanIndex(0, 0)),
                parent=sentence.Sentence("Original Sentence without modifications."),
            ),
            id="Beginning of sentence insertion."
        ),
        pytest.param(
            sentence.Sentence("Original Sentence without modifications."),
            " Text to add at the end.",
            40,
            sentence.Sentence(
                "Original Sentence without modifications. Text to add at the end.", 
                modification=sentence.Modification(" Text to add at the end.", sentence.SpanIndex(40, 40)),
                parent=sentence.Sentence("Original Sentence without modifications."),
            ),
            id="End of sentence replacement."
        )
    ]
)
def test_insert(original: sentence.Sentence, span: str, idx: int, expected: sentence.Sentence):
    output = original.insert(span, idx)
    _assert_equal_sentences(expected, output)

@pytest.mark.parametrize(
    "original,loc,expected",
    [
        pytest.param(
            sentence.Sentence("Original Sentence, text to delete at the middle, without modifications."),
            (17, 48),
            sentence.Sentence(
                "Original Sentence without modifications.", 
                modification=sentence.Modification("", sentence.SpanIndex(17, 48)),
                parent=sentence.Sentence("Original Sentence, text to delete at the middle, without modifications."),
            ),
            id="Middle of sentence deletion."
        ),
        pytest.param(
            sentence.Sentence("Text to delete at the beginning. Original Sentence without modifications."),
            (0, 33),
            sentence.Sentence(
                "Original Sentence without modifications.", 
                modification=sentence.Modification("", sentence.SpanIndex(0, 33)),
                parent=sentence.Sentence("Text to delete at the beginning. Original Sentence without modifications."),
            ),
            id="Beginning of sentence deletion."
        ),
        pytest.param(
            sentence.Sentence("Original Sentence without modifications. Text to delete at the end."),
            (40, 67),
            sentence.Sentence(
                "Original Sentence without modifications.", 
                modification=sentence.Modification("", sentence.SpanIndex(40, 67)),
                parent=sentence.Sentence("Original Sentence without modifications. Text to delete at the end."),
            ),
            id="End of sentence deletion."
        )
    ]
)
def test_deletion(original: sentence.Sentence, loc: sentence.SpanIndexLike, expected: sentence.Sentence):
    output = original.delete(loc)
    _assert_equal_sentences(expected, output)


@pytest.mark.parametrize(
    "original,span,loc,expected",
    [
        pytest.param(
            sentence.Sentence("Original Sentence without modifications."),
            ", text to replace without,",
            (17, 25),
            sentence.Sentence(
                "Original Sentence, text to replace without, modifications.", 
                modification=sentence.Modification(", text to replace without,", sentence.SpanIndex(17, 25)),
                parent=sentence.Sentence("Original Sentence without modifications."),
            ),
            id="Middle of sentence replacement."
        ),
        pytest.param(
            sentence.Sentence("Original Sentence without modifications."),
            "Text to replace Original.",
            (0, 8),
            sentence.Sentence(
                "Text to replace Original. Sentence without modifications.", 
                modification=sentence.Modification("Text to replace Original.", sentence.SpanIndex(0, 8)),
                parent=sentence.Sentence("Original Sentence without modifications."),
            ),
            id="Beginning of sentence replacement."
        ),
        pytest.param(
            sentence.Sentence("Original Sentence without modifications."),
            ". Text to replace modifications.",
            (25, 40),
            sentence.Sentence(
                "Original Sentence without. Text to replace modifications.", 
                modification=sentence.Modification(". Text to replace modifications.", sentence.SpanIndex(25, 40)),
                parent=sentence.Sentence("Original Sentence without modifications."),
            ),
            id="End of sentence replacement."
        )
    ]
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
    assert expected.modification == actual.modification
    if expected.parent is None:
        assert actual.parent is None
    else:
        assert actual.parent is not None
        _assert_equal_sentences(expected.parent, actual.parent)