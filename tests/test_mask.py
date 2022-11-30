import pytest

from smaug import mask


@pytest.mark.parametrize(
    "docs,intervals,func,expected",
    [
        pytest.param(
            "Test string with some words",
            mask.MaskingIntervals.from_list([(0, 4), (10, 15)]),
            lambda _: "<mask>",
            "<mask> strin<mask>h some words",
            id="single sentence with string mask",
        ),
        pytest.param(
            "Test string with some words",
            mask.MaskingIntervals.from_list([(0, 4), (10, 15)]),
            lambda idx: f"<mask-{idx}>",
            "<mask-0> strin<mask-1>h some words",
            id="single sentence with masking function",
        ),
    ],
)
def test_mask_util(docs, intervals, func, expected):
    output = mask.mask_intervals(docs, intervals, func)
    assert expected == output


@pytest.mark.parametrize(
    "text,func,expected",
    [
        pytest.param(
            "Test with 1.23,22 number and another 1.220.",
            lambda _: "<mask>",
            "Test with <mask> number and another <mask>.",
            id="single sentence with string mask",
        ),
        pytest.param(
            "Test with .21234 number and another 2312234.",
            lambda idx: f"<mask-{idx}>",
            "Test with <mask-0> number and another <mask-1>.",
            id="single sentence with masking function",
        ),
    ],
)
def test_number_mask(text, func, expected):
    output = mask.mask_numbers(text, func)
    assert expected == output


@pytest.mark.parametrize(
    "text,func,expected_opts",
    [
        pytest.param(
            "Test with 1.23,22 number and another 1.220.",
            lambda _: "<mask>",
            [
                "Test with <mask> number and another 1.220.",
                "Test with 1.23,22 number and another <mask>.",
            ],
            id="single sentence with string mask",
        ),
        pytest.param(
            "Test with .21234 number and another 1.220.",
            lambda idx: f"<mask-{idx}>",
            [
                "Test with <mask-0> number and another 1.220.",
                "Test with .21234 number and another <mask-0>.",
            ],
            id="single sentence with masking function",
        ),
    ],
)
def test_number_mask_max(text, func, expected_opts):
    output = mask.mask_numbers(text, func, max_masks=1)
    assert output in expected_opts


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
    output = mask.mask_random_replace(text, func, p=0.5)
    print(output)
    t_splits = text.split()
    o_splits = output.split()
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
def test_random_insert_mask(text, func):
    output = mask.mask_random_insert(text, func, p=0.5)
    t_splits = text.split()
    o_splits = output.split()
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
