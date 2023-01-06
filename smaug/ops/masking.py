import functools
import numpy as np


from smaug import ops
from smaug.broadcast import broadcast_data
from smaug.core import Data, DataLike, Sentence, SentenceLike, SpanIndexLike
from smaug.frozen import frozenlist
from smaug.promote import promote_to_data, promote_to_sentence, promote_to_span_index

from typing import Callable, Optional, Tuple

MaskFunction = Callable[[int], str]
"""Retrieves the ith mask token given i.

For methods where masks in the same sentences are different, this function
will be called with 0, 1, 2, ... and should return the 1st, 2nd, 3rd, ... masks.

For methods that do not distinguish masks, this function should always return
the same value.

Args:
    i: mask index.

Returns:
    mask token to insert
"""


def mask_intervals(
    text: DataLike[SentenceLike],
    intervals: DataLike[frozenlist[SpanIndexLike]],
    func: MaskFunction,
) -> Data[Sentence]:
    """Masks a sentence according to intervals.

    Mask the given sentence according to the specified intervals. The characters
    in the specified intervals are replaced by the mask token.

    Args:
        text: text to mask.
        intervals: intervals to mask. Each interval should specify the
        (start, end) to index the sentence.
        func: masking function to mask the intervals.

    Returns:
        Masked text according to the given intervals.
    """

    text = promote_to_data(text)
    intervals = promote_to_data(intervals)

    text, intervals = broadcast_data(text, intervals)

    sentences = map(promote_to_sentence, text)

    return Data(
        _mask_sentence_intervals(s, i, func) for s, i in zip(sentences, intervals)
    )


def _mask_sentence_intervals(
    sentence: Sentence,
    intervals: frozenlist[SpanIndexLike],
    func: MaskFunction,
) -> Sentence:

    if len(intervals) == 0:
        return sentence

    mask_idx = len(intervals) - 1
    # Go through intervals in reverse order as modifying
    # the sentence shifts all intervals greater than the
    # current.
    for interval in sorted(intervals, reverse=True):
        mask = func(mask_idx)
        span_index = promote_to_span_index(interval)
        sentence = ops.replace(sentence, mask, span_index)
        mask_idx -= 1

    return sentence


def mask_detections(
    text: DataLike[SentenceLike],
    detect_func: Callable[[DataLike[SentenceLike]], Data[frozenlist[Tuple[int, int]]]],
    mask_func: MaskFunction,
    rng: np.random.Generator,
    p: float = 1,
    max_masks: Optional[int] = None,
) -> Data[Sentence]:
    """Masks the detected spans in a given text.

    Args:
        text: Text to apply the masks.
        detect_func: Function to detect possible spans to mask.
        mask_func: Masking function to apply.
        rng: Numpy random generator to use.
        p: Probability of applying a mask to a given detection.
        max_masks: Maximum masks to apply. If not specified all
        detections will be masked.

    Returns:
        Masked text.
    """
    text = promote_to_data(text)
    mask_sentence_func = functools.partial(
        _mask_sentence_detections,
        detect_func=detect_func,
        mask_func=mask_func,
        rng=rng,
        p=p,
        max_masks=max_masks,
    )
    sentences = map(promote_to_sentence, text)
    return Data(mask_sentence_func(s) for s in sentences)


def _mask_sentence_detections(
    text: Sentence,
    detect_func: Callable[[DataLike[SentenceLike]], Data[frozenlist[Tuple[int, int]]]],
    mask_func: MaskFunction,
    rng: np.random.Generator,
    p: float = 1,
    max_masks: Optional[int] = None,
) -> Sentence:
    if p == 0:
        return text

    detections = detect_func(text).item()

    if p != 1:
        detections = filter(lambda _: rng.random() < p, detections)

    if max_masks:
        detections = list(detections)
        if len(detections) > max_masks:
            detections = rng.choice(detections, max_masks, replace=False)

    return mask_intervals(text, [frozenlist(detections)], mask_func).item()


def mask_random_replace(
    text: DataLike[SentenceLike],
    func: MaskFunction,
    rng: np.random.Generator,
    p: float = 1,
) -> Data[Sentence]:
    """Randomly replaces words for masks.

    Args:
        text: Text to apply the masks.
        func: Masking function to apply.
        rng: Numpy random generator to use.
        p: Probability of replacing a word by a mask.

    Returns:
        Data[str]: masked text.
    """
    text = promote_to_data(text)
    mask_sentence_func = functools.partial(
        _mask_sentence_random_replace,
        func=func,
        rng=rng,
        p=p,
    )
    sentences = map(promote_to_sentence, text)
    return Data(mask_sentence_func(s) for s in sentences)


def _mask_sentence_random_replace(
    sentence: Sentence,
    func: MaskFunction,
    rng: np.random.Generator,
    p: float = 1,
) -> Sentence:
    def next_word_delim(start: int):
        # Try to find next space
        word_delim_idx = ops.find(sentence, " ", start=start)
        if word_delim_idx == -1:
            # If not space, then we are at the last word
            # and return the remaining sentence.
            word_delim_idx = len(sentence)
        return word_delim_idx

    mask_idx = 0
    curr_idx = 0
    while curr_idx < len(sentence):
        word_delim_idx = next_word_delim(curr_idx)
        if rng.random() < p:
            mask = func(mask_idx)
            sentence = ops.replace(sentence, mask, (curr_idx, word_delim_idx))
            mask_idx += 1
            curr_idx += len(mask) + 1
        else:
            curr_idx = word_delim_idx + 1
    return sentence


def mask_poisson_spans(
    text: DataLike[SentenceLike],
    func: MaskFunction,
    rng: np.random.Generator,
) -> Data[Sentence]:
    """Masks spans of text with sizes following a poisson distribution.

    Args:
        text: Text to mask.
        func: Mask function to apply.
        rng: Numpy random generator to use.

    Returns:
        Masked text.
    """
    text = promote_to_data(text)
    mask_sentence_func = functools.partial(
        _mask_poisson_spans,
        func=func,
        rng=rng,
    )
    sentences = map(promote_to_sentence, text)
    return Data(mask_sentence_func(s) for s in sentences)


def _mask_poisson_spans(
    text: Sentence, func: MaskFunction, rng: np.random.Generator
) -> Sentence:
    # Add plus 1 to indexes as they should index the charcter next
    # to the word limit.
    spaces = [i + 1 for i, c in enumerate(text.value) if c == " "]
    word_starts = [0] + spaces

    found = False
    while not found:
        num_masked_words = rng.poisson()
        start_word_idx = rng.choice(len(word_starts), 1)[0]
        if start_word_idx + num_masked_words <= len(word_starts):
            found = True

    start_idx = word_starts[start_word_idx]
    # We are masking until the end of the sentence.
    if start_word_idx + num_masked_words == len(word_starts):
        end_idx = len(text)
    # We are inserting words.
    elif num_masked_words == 0:
        end_idx = start_idx
    # We are masking words in the middle of the sentence.
    else:
        end_idx = word_starts[start_word_idx + num_masked_words] - 1

    # Only add space if inserting words. Otherwise, use available spaces.
    span = f"{func(0)} " if num_masked_words == 0 else func(0)
    return ops.replace(text, span, (start_idx, end_idx))


def mask_random_insert(
    text: DataLike[SentenceLike],
    func: MaskFunction,
    rng: np.random.Generator,
    p: float = 0.2,
    max_masks: Optional[int] = None,
) -> Data[Sentence]:
    """Inserts masks between random words in the text.

    Args:
        text: Text to apply the masks.
        func: Masking function to apply.
        rng: Numpy random generator to use.
        p: Probability of inserting a mask between two words.
        max_masks: Maximum masks to apply. If not specified all
        regular expression matches will be masked.

    Returns:
        Masked text.
    """
    text = promote_to_data(text)
    mask_sentence_func = functools.partial(
        _mask_sentence_random_insert,
        func=func,
        rng=rng,
        p=p,
        max_masks=max_masks,
    )
    sentences = map(promote_to_sentence, text)
    return Data(mask_sentence_func(s) for s in sentences)


def _mask_sentence_random_insert(
    sentence: Sentence,
    func: MaskFunction,
    rng: np.random.Generator,
    p: float = 0.2,
    max_masks: Optional[int] = None,
) -> Sentence:

    if len(sentence) == 0:
        if rng.random() < p:
            sentence = ops.insert(sentence, func(0), 0)
        return sentence

    after_spaces = [i + 1 for i, c in enumerate(sentence.value) if c == " "]
    # Possible indexes where to start mask.
    possible_mask_starts = np.array([0] + after_spaces + [len(sentence)])

    mask_idxs = rng.choice([False, True], size=len(possible_mask_starts), p=(1 - p, p))
    (true_idxs,) = np.nonzero(mask_idxs)
    if max_masks is not None and len(true_idxs) > max_masks:
        true_idxs = rng.choice(true_idxs, size=max_masks)
        mask_idxs = np.full_like(mask_idxs, False)
        mask_idxs[true_idxs] = True

    mask_start = possible_mask_starts[mask_idxs]

    mask_idx = len(mask_start) - 1
    for idx in reversed(mask_start):
        mask = func(mask_idx)
        # Insert space before unless we are at the beginning, where we insert
        # a space after the mask.
        insert = f"{mask} " if idx != len(sentence) else f" {mask}"
        sentence = ops.insert(sentence, insert, idx)
        mask_idx -= 1

    return sentence
