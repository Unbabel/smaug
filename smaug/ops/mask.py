import dataclasses
import functools
import io
import itertools
import numpy as np
import re

from smaug import _itertools
from smaug import core
from smaug import sentence

from typing import Callable, Iterable, List, Optional, Tuple

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


@dataclasses.dataclass(frozen=True, eq=True, order=True)
class MaskingInterval:
    """Specifies the start and end indexes to be masked in a given string."""

    start: int
    end: int

    def __post_init__(self):
        if self.start < 0:
            raise ValueError(f"'start' must be positive but is {self.start}.")
        if self.end < 0:
            raise ValueError(f"'end' must be positive but is {self.end}.")
        if self.end < self.start:
            msg = f"'end' must be greater or equal to 'start': start={self.start}, end={self.end}"
            raise ValueError(msg)

    def encloses(self, other: "MaskingInterval") -> bool:
        """Verifies whether this interval totally encloses the other.

        If an interval A encloses and Interval B, then:
        A.start  B.start   B.end    A.end
        ---|--------|--------|--------|---
        """
        return self.start <= other.start <= other.end <= self.end

    def partial_overlaps(self, other: "MaskingInterval") -> bool:
        """Verifies whether this interval partially overlaps the other.

        If an interval A partially overlaps Interval B, then:
        A.start  B.start   A.end    B.end
        ---|--------|--------|--------|---
        or
        B.start  A.start   B.end    A.end
        ---|--------|--------|--------|---
        """
        return (
            self.start <= other.start <= self.end <= other.end
            or other.start <= self.start <= other.end <= self.end
        )

    def intersects(self, other: "MaskingInterval") -> bool:
        return (
            self.encloses(other)
            or other.encloses(self)
            or self.partial_overlaps(other)
            or other.partial_overlaps(self)
        )


class MaskingIntervals:
    """Specifies all masking intervals to be masked in a given string."""

    def __init__(self, *intervals: MaskingInterval) -> None:
        self._intervals = list(_itertools.unique_everseen(intervals))

        for i1, i2 in itertools.combinations(self._intervals, 2):
            if i1.intersects(i2):
                raise ValueError(f"Intervals {i1} and {i2} intersect.")

    def sorted(self, reverse: bool = False) -> "MaskingIntervals":
        new_intervals = list(self._intervals)
        new_intervals.sort(reverse=reverse)
        return MaskingIntervals(*new_intervals)

    @classmethod
    def from_list(cls, intervals: List[Tuple[int, int]]) -> "MaskingIntervals":
        mapped = (MaskingInterval(s, e) for s, e in intervals)
        return cls(*mapped)

    def as_ndarray(self) -> np.ndarray:
        return np.array([(i.start, i.end) for i in self._intervals])

    def __repr__(self) -> str:
        intervals_repr = ", ".join(f"({i.start}, {i.end})" for i in self._intervals)
        return f"MaskingIntervals({intervals_repr})"

    def __getitem__(self, idx) -> MaskingInterval:
        return self._intervals[idx]

    def __len__(self):
        return len(self._intervals)


def mask_intervals(
    text: core.DataLike[sentence.SentenceLike],
    intervals: core.DataLike[MaskingIntervals],
    func: MaskFunction,
) -> core.Data[sentence.Sentence]:
    """Masks a sentence according to intervals.

    Mask the given sentence according to the specified intervals. The characters
    in the specified intervals are replaced by the mask token.

    Args:
        text: text to mask.
        intervals: intervals to mask. Each interval should specify the
        start:end to index the sentence.
        func: masking function to mask the intervals.

    Returns:
        Masked text according to the given intervals.
    """

    text = core.promote_to_data(text)
    intervals = core.promote_to_data(intervals)

    text, intervals = core.broadcast_data(text, intervals)

    sentences = map(sentence.promote_to_sentence, text)

    return core.Data(
        _mask_sentence_intervals(s, i, func) for s, i in zip(sentences, intervals)
    )


def _mask_sentence_intervals(
    sentence: sentence.Sentence,
    intervals: MaskingIntervals,
    func: MaskFunction,
) -> sentence.Sentence:

    if len(intervals) == 0:
        return sentence

    mask_idx = len(intervals) - 1
    # Go through intervals in reverse order as modifing
    # the sentence shifts all intervals greater then the
    # current.
    for interval in intervals.sorted(reverse=True):
        mask = func(mask_idx)
        sentence = sentence.replace(mask, (interval.start, interval.end))
        mask_idx -= 1

    return sentence


def mask_named_entities(
    text: core.DataLike[sentence.SentenceLike],
    ner_func: Callable[[core.DataLike[str]], core.Data],
    mask_func: MaskFunction,
    rng: np.random.Generator,
    filter_entities: Optional[Iterable[str]] = None,
    p: float = 1,
    max_masks: Optional[int] = None,
) -> core.Data[sentence.Sentence]:
    """Masks the named entities in a given text.

    Args:
        text: Text to apply the masks.
        ner_func: Function to detect named entities.
        mask_func: Masking function to apply.
        rng: Numpy random generator to use.
        filter_entities: Named entity tags to consider.
        p: Probability of applying a mask to a given named entity.
        max_masks: Maximum masks to apply. If not specified all
        found named entities will be masked.

    Returns:
        Masked text.
    """
    text = core.promote_to_data(text)
    mask_sentence_func = functools.partial(
        _mask_sentence_named_entities,
        ner_func=ner_func,
        mask_func=mask_func,
        rng=rng,
        filter_entities=filter_entities,
        p=p,
        max_masks=max_masks,
    )
    sentences = map(sentence.promote_to_sentence, text)
    return core.Data(mask_sentence_func(s) for s in sentences)


def _mask_sentence_named_entities(
    text: sentence.Sentence,
    ner_func: Callable[[core.DataLike[str]], core.Data],
    mask_func: MaskFunction,
    rng: np.random.Generator,
    filter_entities: Optional[Iterable[str]] = None,
    p: float = 1,
    max_masks: Optional[int] = None,
) -> sentence.Sentence:
    if p == 0:
        return text

    # No filter
    filter_entities_func = lambda ent: True
    if filter_entities is not None:
        unique_entities = set(filter_entities)
        filter_entities_func = lambda ent: ent.type in unique_entities

    text_w_ner = ner_func(text.value).item()

    detected_entities = filter(filter_entities_func, text_w_ner.entities)
    if p != 1:
        detected_entities = filter(lambda _: rng.random() <= p, detected_entities)

    if max_masks:
        detected_entities = list(detected_entities)
        if len(detected_entities) > max_masks:
            detected_entities = rng.choice(detected_entities, max_masks, replace=False)

    intervals_iter = (
        MaskingInterval(ent.start_char, ent.end_char) for ent in detected_entities
    )
    return mask_intervals(text, MaskingIntervals(*intervals_iter), mask_func).item()


_DEFAULT_NUMBERS_REGEX = re.compile(r"[-+]?\.?(\d+[.,])*\d+")


def mask_numbers(
    text: core.DataLike[sentence.SentenceLike],
    func: MaskFunction,
    rng: np.random.Generator,
    max_masks: Optional[int] = None,
) -> core.Data[sentence.Sentence]:
    """Masks the numbers in a given sentence according to regular expressions.

    Args:
        text: Text to apply the masks.
        func: Masking function to apply.
        rng: Numpy random generator to use.
        max_masks: Maximum masks to apply. If not specified all regular
        expression matches will be masked.

    Returns:
        Masked text.
    """
    return mask_regex(text, func, _DEFAULT_NUMBERS_REGEX, rng, max_masks)


def mask_regex(
    text: core.DataLike[sentence.SentenceLike],
    func: MaskFunction,
    regex: re.Pattern,
    rng: np.random.Generator,
    max_masks: Optional[int] = None,
) -> core.Data[sentence.Sentence]:
    """Masks text spans that match a given regular expression.

    Args:
        text: Text to apply the masks.
        func: Masking function to apply.
        regex: Regular expression to match.
        rng: Numpy random generator to use.
        max_masks: Maximum masks to apply. If not specified all
        regular expression matches will be masked.

    Returns:
        Masked text.
    """
    text = core.promote_to_data(text)
    mask_sentence_func = functools.partial(
        _mask_sentence_regex,
        func=func,
        regex=regex,
        rng=rng,
        max_masks=max_masks,
    )
    sentences = map(sentence.promote_to_sentence, text)
    return core.Data(mask_sentence_func(s) for s in sentences)


def _mask_sentence_regex(
    text: sentence.Sentence,
    func: MaskFunction,
    regex: re.Pattern,
    rng: np.random.Generator,
    max_masks: Optional[int] = None,
) -> sentence.Sentence:
    matches = regex.finditer(text.value)
    if max_masks:
        matches = list(matches)
        if len(matches) > max_masks:
            matches = rng.choice(matches, max_masks, replace=False)
    intervals_iter = (MaskingInterval(*m.span()) for m in matches)
    return mask_intervals(text, MaskingIntervals(*intervals_iter), func).item()


def mask_random_replace(
    text: core.DataLike[sentence.SentenceLike],
    func: MaskFunction,
    rng: np.random.Generator,
    p: float = 1,
) -> core.Data[sentence.Sentence]:
    """Randomly replaces words for masks.

    Args:
        text: Text to apply the masks.
        func: Masking function to apply.
        rng: Numpy random generator to use.
        p: Probability of replacing a word by a mask.

    Returns:
        core.Data[str]: masked text.
    """
    text = core.promote_to_data(text)
    mask_sentence_func = functools.partial(
        _mask_sentence_random_replace,
        func=func,
        rng=rng,
        p=p,
    )
    sentences = map(sentence.promote_to_sentence, text)
    return core.Data(mask_sentence_func(s) for s in sentences)


def _mask_sentence_random_replace(
    sentence: sentence.Sentence,
    func: MaskFunction,
    rng: np.random.Generator,
    p: float = 1,
) -> sentence.Sentence:
    def next_word_delim(start: int):
        # Try to find next space
        word_delim_idx = sentence.find(" ", start=start)
        if word_delim_idx == -1:
            # If not space, then we are at the last word
            # and return the remainig sentence.
            word_delim_idx = len(sentence)
        return word_delim_idx

    mask_idx = 0
    curr_idx = 0
    while curr_idx < len(sentence):
        word_delim_idx = next_word_delim(curr_idx)
        if rng.random() < p:
            mask = func(mask_idx)
            sentence = sentence.replace(mask, (curr_idx, word_delim_idx))
            mask_idx += 1
            curr_idx += len(mask) + 1
        else:
            curr_idx = word_delim_idx + 1
    return sentence


def mask_poisson_spans(
    text: core.DataLike[sentence.SentenceLike],
    func: MaskFunction,
    rng: np.random.Generator,
) -> core.Data[sentence.Sentence]:
    """Masks spans of text with sizes following a poisson distribution.

    Args:
        text: Text to mask.
        func: Mask function to apply.
        rng: Numpy random generator to use.

    Returns:
        Masked text.
    """
    text = core.promote_to_data(text)
    mask_sentence_func = functools.partial(
        _mask_poisson_spans,
        func=func,
        rng=rng,
    )
    sentences = map(sentence.promote_to_sentence, text)
    return core.Data(mask_sentence_func(s) for s in sentences)


def _mask_poisson_spans(
    text: sentence.Sentence, func: MaskFunction, rng: np.random.Generator
) -> sentence.Sentence:
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
    # This can happed when we chose a word to start and mask the
    # remainder of the sentence.
    end_idx = (
        len(text)
        if start_word_idx + num_masked_words == len(word_starts)
        else word_starts[start_word_idx + num_masked_words]
    )

    # If adding in the last position, dont add final space.
    span = f"{func(0)} " if end_idx != len(text) else f"{func(0)}"
    return text.replace(span, (start_idx, end_idx))


def mask_random_insert(
    text: core.DataLike[sentence.SentenceLike],
    func: MaskFunction,
    rng: np.random.Generator,
    p: float = 0.2,
    max_masks: Optional[int] = None,
) -> core.Data[sentence.Sentence]:
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
    text = core.promote_to_data(text)
    mask_sentence_func = functools.partial(
        _mask_sentence_random_insert,
        func=func,
        rng=rng,
        p=p,
        max_masks=max_masks,
    )
    sentences = map(sentence.promote_to_sentence, text)
    return core.Data(mask_sentence_func(s) for s in sentences)


def _mask_sentence_random_insert(
    sentence: sentence.Sentence,
    func: MaskFunction,
    rng: np.random.Generator,
    p: float = 0.2,
    max_masks: Optional[int] = None,
) -> sentence.Sentence:

    if len(sentence) == 0:
        if rng.random() < p:
            sentence = sentence.insert(func(0), 0)
        return sentence

    spaces = [i for i, c in enumerate(sentence.value) if c == " "]
    # Possible indexes where to start mask.
    possible_mask_starts = np.array([0] + spaces + [len(sentence)])

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
        insert = f" {mask}" if idx != 0 else f"{mask} "
        sentence = sentence.insert(insert, idx)
        mask_idx -= 1

    return sentence
