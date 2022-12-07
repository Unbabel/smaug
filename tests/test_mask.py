import numpy as np
import pytest

from smaug import core
from smaug.ops import mask


@pytest.mark.parametrize(
    "docs,intervals,func,expected",
    [
        pytest.param(
            "Test string with some words",
            mask.MaskingIntervals.from_list([(0, 4), (10, 15)]),
            lambda _: "<mask>",
            core.Data(["<mask> strin<mask>h some words"]),
            id="single sentence with string mask",
        ),
        pytest.param(
            ["Test string with some words", "2nd string to mask."],
            mask.MaskingIntervals.from_list([(0, 4), (10, 15)]),
            lambda _: "<mask>",
            core.Data(["<mask> strin<mask>h some words", "<mask>string<mask>ask."]),
            id="multiple sentences with string mask",
        ),
        pytest.param(
            "Test string with some words",
            mask.MaskingIntervals.from_list([(0, 4), (10, 15)]),
            lambda idx: f"<mask-{idx}>",
            core.Data(["<mask-0> strin<mask-1>h some words"]),
            id="single sentence with masking function",
        ),
        pytest.param(
            ["Test string with some words", "2nd string to mask."],
            mask.MaskingIntervals.from_list([(0, 4), (10, 15)]),
            lambda idx: f"<mask-{idx}>",
            core.Data(
                ["<mask-0> strin<mask-1>h some words", "<mask-0>string<mask-1>ask."]
            ),
            id="multiple sentences with masking function",
        ),
    ],
)
def test_mask_intervals(docs, intervals, func, expected):
    output = mask.mask_intervals(docs, intervals, func)
    assert isinstance(output, core.Data)
    assert len(expected) == len(output)
    for e, p in zip(expected, output):
        assert e == p


@pytest.mark.parametrize(
    "text,func,expected",
    [
        pytest.param(
            "Test with 1.23,22 number and another 1.220.",
            lambda _: "<mask>",
            core.Data(["Test with <mask> number and another <mask>."]),
            id="single sentence with string mask",
        ),
        pytest.param(
            "Test with .21234 number and another 2312234.",
            lambda idx: f"<mask-{idx}>",
            core.Data(["Test with <mask-0> number and another <mask-1>."]),
            id="single sentence with masking function",
        ),
    ],
)
def test_number_mask(text, func, expected):
    output = mask.mask_numbers(text, func, np.random.default_rng())
    assert isinstance(output, core.Data)
    assert len(expected) == len(output)
    for e, p in zip(expected, output):
        assert e == p


@pytest.mark.parametrize(
    "text,func,expected_opts",
    [
        pytest.param(
            "Test with 1.23,22 number and another 1.220.",
            lambda _: "<mask>",
            [
                core.Data(["Test with <mask> number and another 1.220."]),
                core.Data(["Test with 1.23,22 number and another <mask>."]),
            ],
            id="single sentence with string mask",
        ),
        pytest.param(
            "Test with .21234 number and another 1.220.",
            lambda idx: f"<mask-{idx}>",
            [
                core.Data(["Test with <mask-0> number and another 1.220."]),
                core.Data(["Test with .21234 number and another <mask-0>."]),
            ],
            id="single sentence with masking function",
        ),
    ],
)
def test_number_mask_max(text, func, expected_opts):
    def matches_func(expected, output):
        return (
            isinstance(output, core.Data)
            and len(expected) == len(output)
            and all(e == p for e, p in zip(expected, output))
        )

    output = mask.mask_numbers(text, func, np.random.default_rng(), max_masks=1)
    assert any(matches_func(e, output) for e in expected_opts)


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
    output = mask.mask_random_replace(text, func, np.random.default_rng(), p=0.5)

    t_splits = text.split()
    assert isinstance(output, core.Data)
    assert len(output) == 1
    o_splits = output.item().split()
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

    output = mask.mask_poisson_spans(text, func, np.random.default_rng())

    assert isinstance(output, core.Data)
    assert len(output) == 1

    o_splits = output.item().split()
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
    assert rev_idx != -1

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
    output = mask.mask_random_insert(text, func, np.random.default_rng(), p=0.5)

    assert isinstance(output, core.Data)
    assert len(output) == 1
    t_splits = text.split()
    o_splits = output.item().split()
    assert len(t_splits) <= len(o_splits)

    mask_idx = 0
    t_idx = 0
    for o_word in o_splits:
        t_word = t_splits[t_idx]
        # No mask was inserted. Move d_idx forward.
        if t_word == o_word:
            t_idx += 1
        # Mask inserted. Verify it is correct.
        else:
            assert func(mask_idx) == o_word
            mask_idx += 1
    # All original words were matched.
    assert t_idx == len(t_splits)
