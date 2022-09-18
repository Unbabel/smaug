import pytest

from smaug._itertools import ResetableIterator
from smaug.mask import MaskIterator
from smaug.mask import numerical
from smaug.mask import random
from smaug.mask import func


class MaskingFunction(ResetableIterator):
    def __init__(self):
        self.__counter = 0

    def __next__(self):
        mask = f"<mask-{self.__counter}>"
        self.__counter += 1
        return mask

    def reset(self):
        self.__counter = 0


@pytest.mark.parametrize(
    "docs,intervals,pattern,expected",
    [
        pytest.param(
            "Test string with some words",
            [(0, 4), (10, 15)],
            "<mask>",
            "<mask> strin<mask>h some words",
            id="single sentence with string mask",
        ),
        pytest.param(
            ["First test string to be masked", "Second test string to be masked"],
            [[(1, 5), (9, 17)], [(7, 8), (0, 4)]],
            "<mask>",
            [
                "F<mask> tes<mask> to be masked",
                "<mask>nd <mask>est string to be masked",
            ],
            id="multiple sentences with string mask",
        ),
        pytest.param(
            "Test string with some words",
            [(0, 4), (10, 15)],
            MaskingFunction(),
            "<mask-0> strin<mask-1>h some words",
            id="single sentence with masking function",
        ),
        pytest.param(
            ["First test string to be masked", "Second test string to be masked"],
            [[(1, 5), (9, 17)], [(6, 7), (0, 4)]],
            MaskingFunction(),
            [
                "F<mask-0> tes<mask-1> to be masked",
                "<mask-0>nd<mask-1>test string to be masked",
            ],
            id="multiple sentences with masking function",
        ),
    ],
)
def test_mask_util(docs, intervals, pattern, expected):
    output = func.mask(docs, intervals, pattern)
    assert expected == output


@pytest.mark.parametrize(
    "docs,pattern,expected",
    [
        pytest.param(
            "Test with 1 number and another 1.220.",
            "<mask>",
            "Test with <mask> number and another <mask>.",
            id="single sentence with string mask",
        ),
        pytest.param(
            [
                "First test with 1.23,22 and 1234.1223",
                "Second test has .21234 and 2312234.",
            ],
            "<mask>",
            [
                "First test with <mask> and <mask>",
                "Second test has <mask> and <mask>.",
            ],
            id="multiple sentences with string mask",
        ),
        pytest.param(
            "Test with 1 number and another 1.220.",
            MaskingFunction(),
            "Test with <mask-0> number and another <mask-1>.",
            id="single sentence with masking function",
        ),
        pytest.param(
            [
                "First test with 1.23,22 and 1234.1223",
                "Second test has .21234 and 2312234.",
            ],
            MaskingFunction(),
            [
                "First test with <mask-0> and <mask-1>",
                "Second test has <mask-0> and <mask-1>.",
            ],
            id="multiple sentences with masking function",
        ),
    ],
)
def test_number_mask(docs, pattern, expected):
    number_mask = numerical.Number(pattern)
    output = number_mask(docs)
    assert expected == output


@pytest.mark.parametrize(
    "docs,pattern,expected_opts",
    [
        pytest.param(
            "Test with 1 number and another 1.220.",
            "<mask>",
            [
                "Test with <mask> number and another 1.220.",
                "Test with 1 number and another <mask>.",
            ],
            id="single sentence with string mask",
        ),
        pytest.param(
            [
                "First test with 1.23,22 and 1234.1223",
                "Second test has .21234 only.",
            ],
            "<mask>",
            [
                [
                    "First test with 1.23,22 and <mask>",
                    "First test with <mask> and 1234.1223",
                ],
                [
                    "Second test has <mask> only.",
                ],
            ],
            id="multiple sentences with string mask",
        ),
        pytest.param(
            "Test with 1 number and another 1.220.",
            MaskingFunction(),
            [
                "Test with <mask-0> number and another 1.220.",
                "Test with 1 number and another <mask-0>.",
            ],
            id="single sentence with masking function",
        ),
        pytest.param(
            [
                "First test with 1.23,22 and 1234.1223",
                "Second test has .21234 only.",
            ],
            MaskingFunction(),
            [
                [
                    "First test with 1.23,22 and <mask-0>",
                    "First test with <mask-0> and 1234.1223",
                ],
                [
                    "Second test has <mask-0> only.",
                ],
            ],
            id="multiple sentences with masking function",
        ),
    ],
)
def test_number_mask_max(docs, pattern, expected_opts):
    number_mask = numerical.Number(pattern, max_mask=1)
    output = number_mask(docs)
    if isinstance(docs, str):
        output = [output]
        expected_opts = [expected_opts]
    for o, opts in zip(output, expected_opts):
        assert o in opts


@pytest.mark.parametrize(
    "docs,pattern",
    [
        pytest.param(
            "Test with 1 number and another 1.220.",
            "<mask>",
            id="single sentence with string mask",
        ),
        pytest.param(
            [
                "First test with 1.23,22 and 1234.1223",
                "Second test has .21234 and 2312234.",
            ],
            "<mask>",
            id="multiple sentences with string mask",
        ),
        pytest.param(
            "Test with 1 number and another 1.220.",
            MaskingFunction(),
            id="single sentence with masking function",
        ),
        pytest.param(
            [
                "First test with 1.23,22 and 1234.1223",
                "Second test has .21234 and 2312234.",
            ],
            MaskingFunction(),
            id="multiple sentences with masking function",
        ),
    ],
)
def test_random_replace_mask(docs, pattern):
    word_mask = random.RandomReplace(pattern, p=0.5)
    output = word_mask(docs)
    if isinstance(docs, str):
        assert isinstance(output, str)
        docs, output = [docs], [output]
    for d, o in zip(docs, output):
        d_splits = d.split()
        o_splits = o.split()
        assert len(d_splits) == len(o_splits)
        mask_iter = MaskIterator(pattern)
        mask_iter.reset()
        for d_word, o_word in zip(d_splits, o_splits):
            # Conditions must be in this order to ensure the
            # next method os only called for the non equal words
            assert d_word == o_word or next(mask_iter) == o_word


@pytest.mark.parametrize(
    "docs,pattern",
    [
        pytest.param(
            "Test with 1 number and another 1.220.",
            "<mask>",
            id="single sentence with string mask",
        ),
        pytest.param(
            [
                "First test with 1.23,22 and 1234.1223",
                "Second test has .21234 and 2312234.",
            ],
            "<mask>",
            id="multiple sentences with string mask",
        ),
        pytest.param(
            "Test with 1 number and another 1.220.",
            MaskingFunction(),
            id="single sentence with masking function",
        ),
        pytest.param(
            [
                "First test with 1.23,22 and 1234.1223",
                "Second test has .21234 and 2312234.",
            ],
            MaskingFunction(),
            id="multiple sentences with masking function",
        ),
    ],
)
def test_random_insert_mask(docs, pattern):
    word_mask = random.RandomInsert(pattern, p=0.5)
    output = word_mask(docs)
    if isinstance(docs, str):
        assert isinstance(output, str)
        docs, output = [docs], [output]
    for d, o in zip(docs, output):
        d_splits = d.split()
        o_splits = o.split()
        assert len(d_splits) <= len(o_splits)
        mask_iter = MaskIterator(pattern)
        mask_iter.reset()
        d_idx = 0
        for o_word in o_splits:
            d_word = d_splits[d_idx]
            # No mask was inserted. Move d_idx forward.
            if d_word == o_word:
                d_idx += 1
            # Mask inserted. Verify it is correct.
            else:
                assert next(mask_iter) == o_word
        # All original words were matched.
        assert d_idx == len(d_splits)
