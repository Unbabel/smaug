import numpy as np
import pytest

from smaug.core import Data
from smaug.frozen import frozenlist
from smaug import ops


@pytest.mark.parametrize(
    "docs,intervals,func,expected",
    [
        pytest.param(
            "Test string with some words",
            [frozenlist([(0, 4), (10, 15)])],
            lambda _: "<mask>",
            Data(["<mask> strin<mask>h some words"]),
            id="single sentence with string mask",
        ),
        pytest.param(
            ["Test string with some words", "2nd string to mask."],
            [frozenlist([(0, 4), (10, 15)])],
            lambda _: "<mask>",
            Data(["<mask> strin<mask>h some words", "<mask>string<mask>ask."]),
            id="multiple sentences with string mask",
        ),
        pytest.param(
            "Test string with some words",
            [frozenlist([(0, 4), (10, 15)])],
            lambda idx: f"<mask-{idx}>",
            Data(["<mask-0> strin<mask-1>h some words"]),
            id="single sentence with masking function",
        ),
        pytest.param(
            ["Test string with some words", "2nd string to mask."],
            [frozenlist([(0, 4), (10, 15)])],
            lambda idx: f"<mask-{idx}>",
            Data(["<mask-0> strin<mask-1>h some words", "<mask-0>string<mask-1>ask."]),
            id="multiple sentences with masking function",
        ),
    ],
)
def test_mask_intervals(docs, intervals, func, expected):
    output = ops.mask_intervals(docs, intervals, func)
    assert isinstance(output, Data)
    assert len(expected) == len(output)
    for e, p in zip(expected, output):
        assert e == p.value


@pytest.mark.parametrize(
    "text,detect_func,mask_func,expected",
    [
        pytest.param(
            "Test with 1.23,22 number and another 1.220.",
            lambda _: Data([frozenlist([(10, 17), (37, 42)])]),
            lambda _: "<mask>",
            Data(["Test with <mask> number and another <mask>."]),
            id="single sentence with string mask",
        ),
        pytest.param(
            "Test with .21234 number and another 2312234.",
            lambda _: Data([frozenlist([(10, 16), (36, 43)])]),
            lambda idx: f"<mask-{idx}>",
            Data(["Test with <mask-0> number and another <mask-1>."]),
            id="single sentence with masking function",
        ),
    ],
)
def test_mask_detections(text, detect_func, mask_func, expected):
    output = ops.mask_detections(text, detect_func, mask_func, np.random.default_rng())
    assert isinstance(output, Data)
    assert len(expected) == len(output)
    for e, p in zip(expected, output):
        assert e == p.value


@pytest.mark.parametrize(
    "text,detect_func,mask_func,expected_opts",
    [
        pytest.param(
            "Test with 1.23,22 number and another 1.220.",
            lambda _: Data([frozenlist([(10, 17), (37, 42)])]),
            lambda _: "<mask>",
            [
                Data(["Test with <mask> number and another 1.220."]),
                Data(["Test with 1.23,22 number and another <mask>."]),
            ],
            id="single sentence with string mask",
        ),
        pytest.param(
            "Test with .21234 number and another 2312234.",
            lambda _: Data([frozenlist([(10, 16), (36, 43)])]),
            lambda idx: f"<mask-{idx}>",
            [
                Data(["Test with <mask-0> number and another 2312234."]),
                Data(["Test with .21234 number and another <mask-0>."]),
            ],
            id="single sentence with masking function",
        ),
    ],
)
def test_mask_detections_max_masks(text, detect_func, mask_func, expected_opts):
    def matches_func(expected):
        return (
            isinstance(output, Data)
            and len(expected) == len(output)
            and all(e == p.value for e, p in zip(expected, output))
        )

    output = ops.mask_detections(
        text, detect_func, mask_func, np.random.default_rng(), max_masks=1
    )
    assert any(matches_func(e) for e in expected_opts)


@pytest.mark.parametrize(
    "text,func",
    [
        pytest.param(
            "Test with 1 number and another 1.220.",
            lambda _: "<mask>",
            id="single sentence with string mask",
        ),
        pytest.param(
            "Test with 1 number and another 1.220.",
            lambda idx: f"<mask-{idx}>",
            id="single sentence with masking function",
        ),
    ],
)
def test_random_replace_mask(text: str, func):
    output = ops.mask_random_replace(text, func, np.random.default_rng(), p=0.5)
    t_splits = text.split()
    assert isinstance(output, Data)
    assert len(output) == 1
    o_splits = output.item().value.split()
    assert len(t_splits) == len(o_splits)

    mask_idx = 0
    for t_word, o_word in zip(t_splits, o_splits):
        # If words differ then a mask should have bee inserted.
        if t_word != o_word:
            assert func(mask_idx) == o_word
            mask_idx += 1


@pytest.mark.parametrize(
    "text,func",
    [
        pytest.param(
            "Test with 1 number and another 1.220.",
            lambda _: "<mask>",
            id="single sentence with string mask",
        ),
        pytest.param(
            "Test with 1 number and another 1.220.",
            lambda idx: f"<mask-{idx}>",
            id="single sentence with masking function",
        ),
    ],
)
def test_mask_poisson_spans(text: str, func):
    def first_mismatch(list1, list2):
        for i in range(min(len(list1), len(list2))):
            if list1[i] != list2[i]:
                return i
        return -1

    output = ops.mask_poisson_spans(text, func, np.random.default_rng())
    assert isinstance(output, Data)
    assert len(output) == 1

    o_splits = output.item().value.split()
    t_splits = text.split()

    num_splits_diff = len(t_splits) - len(o_splits)
    # Maximum one extra word is inserted
    assert num_splits_diff >= -1

    # Index of first mismatch going forward
    fwd_idx = first_mismatch(t_splits, o_splits)
    assert fwd_idx != -1
    # Mismatch must happen on mask
    assert o_splits[fwd_idx] == func(0)

    # Index of first mismatch going backwards.
    # This index only works for reversed splits.
    rev_idx = first_mismatch(t_splits[::-1], o_splits[::-1])
    # Can happen if mask was inserted in the beginning while
    # masking 0 words
    if rev_idx == -1:
        rev_idx = len(o_splits) - 1

    # Rev index considering forward o_splits
    o_rev_idx = len(o_splits) - rev_idx - 1
    # Rev index considering forward t_splits
    t_rev_idx = len(t_splits) - rev_idx - 1

    # Mismatch must happen on mask
    assert o_splits[o_rev_idx] == func(0)

    # Difference in words must be the same as the difference
    # between the indexes.
    assert num_splits_diff == t_rev_idx - fwd_idx


@pytest.mark.parametrize(
    "text,func",
    [
        pytest.param(
            "Test with 1 number and another 1.220.",
            lambda _: "<mask>",
            id="single sentence with string mask",
        ),
        pytest.param(
            "Test with 1 number and another 1.220.",
            lambda idx: f"<mask-{idx}>",
            id="single sentence with masking function",
        ),
    ],
)
def test_random_insert_mask(text, func):
    output = ops.mask_random_insert(text, func, np.random.default_rng(), p=0.5)

    assert isinstance(output, Data)
    assert len(output) == 1
    t_splits = text.split()
    o_splits = output.item().value.split()
    assert len(t_splits) <= len(o_splits)

    mask_idx = 0
    t_idx = 0
    for o_word in o_splits:
        # No mask was inserted. Move d_idx forward.
        if t_idx < len(t_splits) and t_splits[t_idx] == o_word:
            t_idx += 1
        # Mask inserted. Verify it is correct.
        else:
            assert func(mask_idx) == o_word
            mask_idx += 1
    # All original words were matched.
    assert t_idx == len(t_splits)
