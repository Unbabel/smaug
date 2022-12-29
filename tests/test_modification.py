import itertools
import pytest

from smaug import frozen
from smaug import ops
from smaug.core import Modification, ModificationTrace, ModifiedIndices, SpanIndex


@pytest.mark.parametrize(
    "old,new,modification",
    [
        pytest.param(
            'Sentence with "old text to be replaced" in the middle.',
            'Sentence with "replaced new text" in the middle.',
            Modification(
                '"old text to be replaced"',
                '"replaced new text"',
                14,
            ),
            id="Modify the middle of the sentence.",
        ),
        pytest.param(
            '"Sentence with old text to be replaced" in the beginning.',
            '"Sentence with replaced new text" in the beginning.',
            Modification(
                '"Sentence with old text to be replaced"',
                '"Sentence with replaced new text"',
                0,
            ),
            id="Modify the beginning of the sentence.",
        ),
        pytest.param(
            'Sentence with "old text to be replaced in the end".',
            'Sentence with "replaced new text in the end".',
            Modification(
                '"old text to be replaced in the end".',
                '"replaced new text in the end".',
                14,
            ),
            id="Modify the end of the sentence.",
        ),
    ],
)
def test_modification(old: str, new: str, modification: Modification):
    new_output = ops.apply_modification(modification, old)
    assert new == new_output
    old_output = ops.reverse_modification(modification, new_output)
    assert old == old_output


@pytest.mark.parametrize(
    "old,new,trace,expected_indices,expected_spans",
    [
        pytest.param(
            'Original Sentence with "text to modify" and "more text to modify".',
            'Original Sentence with "modified text" and "more modifed text".',
            ModificationTrace.from_modifications(
                Modification('"text to modify"', '"modified text"', 23),
                Modification('"more text to modify"', '"more modifed text"', 43),
            ),
            ModifiedIndices(itertools.chain(range(23, 38), range(43, 62))),
            frozen.frozenlist([SpanIndex(23, 38), SpanIndex(43, 62)]),
            id="No overlap",
        ),
        pytest.param(
            'Original Sentence with "text to modify" and "more text to modify".',
            'Original Sentence with "modified "overlapped text" modifed text".',
            ModificationTrace.from_modifications(
                Modification('"text to modify"', '"modified text"', 23),
                Modification('"more text to modify"', '"more modifed text"', 43),
                Modification('text" and "more', '"overlapped text"', 33),
            ),
            ModifiedIndices(range(23, 64)),
            frozen.frozenlist([SpanIndex(23, 64)]),
            id="With overlap",
        ),
    ],
)
def test_modification_trace(
    old: str,
    new: str,
    trace: ModificationTrace,
    expected_indices: ModifiedIndices,
    expected_spans: frozen.frozenlist[SpanIndex],
):
    new_output = ops.apply_modification_trace(trace, old)
    assert new == new_output
    old_output = ops.reverse_modification_trace(trace, new_output)
    assert old == old_output

    modified_indices = ops.modified_indices_from_trace(trace)
    assert len(expected_indices) == len(modified_indices)

    modified_spans = ops.compress_modified_indices(modified_indices)
    assert len(expected_spans) == len(modified_spans)
    for e, m in zip(expected_spans, modified_spans):
        assert e.start == m.start
        assert e.end == m.end
