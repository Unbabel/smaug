import dataclasses
import io
import itertools
import numpy as np
import re

from smaug import _itertools
from smaug import random

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

    def sorted(self) -> "MaskingIntervals":
        new_intervals = list(self._intervals)
        new_intervals.sort()
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

    def __len__(self):
        return len(self._intervals)


def mask_intervals(sentence: str, intervals: MaskingIntervals, func: MaskFunction):
    """Masks a sentence according to intervals.

    Mask the given sentence according to the specified intervals. The characters
    in the specified intervals are replaced by the mask token.

    Args:
        sentence: sentence to mask.
        intervals: intervals to mask. Each interval should specify this
            start:end to index the sentence.
        func: masking function to mask the intervals.

    Returns:
        A masked document or a list of masked documents according to the given
        intervals.
    """
    if len(intervals) == 0:
        return sentence

    # Compute the chunks of the sentence that are not masked
    # There are three options:
    # 1 - Before the first index
    # 2 - After the last index
    # 3 - Between the end of an interval and the start of the next interval
    #
    # As an example, the following intervals [(1:3),(5:8),(12:13)] is
    # flattened to [1, 3, 5, 8, 12, 13] and the chunks to keep are
    # [[:1], [3:5], [8:12], [13:]].
    chunks = []
    indexes = intervals.sorted().as_ndarray().ravel()
    for interval_idx in range(len(indexes)):
        # First option
        if interval_idx == 0:
            chunks.append(sentence[: indexes[interval_idx]])
        # Second option
        elif interval_idx == len(indexes) - 1:
            chunks.append(sentence[indexes[interval_idx] :])
        # Third option
        elif interval_idx % 2 == 1:
            chunks.append(sentence[indexes[interval_idx] : indexes[interval_idx + 1]])

    buffer = io.StringIO()
    mask_idx = 0
    for i, c in enumerate(chunks):
        buffer.write(c)
        if i < len(chunks) - 1:
            buffer.write(func(mask_idx))
            mask_idx += 1

    return buffer.getvalue()


def mask_named_entities(
    text: str,
    ner_func: Callable,
    mask_func: MaskFunction,
    filter_entities: Optional[Iterable[str]] = None,
    p: float = 1,
    max_masks: Optional[int] = None,
) -> str:
    if p == 0:
        return text

    # No filter
    filter_entities_func = lambda ent: True
    if filter_entities is not None:
        unique_entities = set(filter_entities)
        filter_entities_func = lambda ent: ent.type in unique_entities

    rng = random.numpy_seeded_rng()

    text_w_ner = ner_func(text)

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
    return mask_intervals(text, MaskingIntervals(*intervals_iter), mask_func)


_DEFAULT_NUMBERS_REGEX = re.compile(r"[-+]?\.?(\d+[.,])*\d+")


def mask_numbers(
    text: str,
    func: MaskFunction,
    regex: re.Pattern = _DEFAULT_NUMBERS_REGEX,
    max_masks: Optional[int] = None,
) -> str:
    rng = random.numpy_seeded_rng()
    matches = regex.finditer(text)
    if max_masks:
        matches = list(matches)
        if len(matches) > max_masks:
            matches = rng.choice(matches, max_masks, replace=False)
    intervals_iter = (MaskingInterval(*m.span()) for m in matches)
    return mask_intervals(text, MaskingIntervals(*intervals_iter), func)


def mask_random_replace(text: str, func: MaskFunction, p: float = 1) -> str:
    rng = random.numpy_seeded_rng()
    splits = text.split()
    mask = rng.choice([False, True], size=len(splits), p=(1 - p, p))
    mask_iter = (func(i) for i in itertools.count())
    splits = [next(mask_iter) if m else s for s, m in zip(splits, mask)]
    return " ".join(splits)


def mask_random_insert(
    text: str,
    func: MaskFunction,
    p: float = 0.2,
    max_masks: Optional[int] = None,
) -> str:
    rng = random.numpy_seeded_rng()

    splits = text.split()
    # Not enought splits to insert mask.
    if len(splits) < 2:
        return text

    mask_idxs = rng.choice([False, True], size=len(splits) - 1, p=(p, 1 - p))
    (true_idxs,) = np.nonzero(mask_idxs)
    if max_masks is not None and len(true_idxs) > max_masks:
        true_idxs = rng.choice(true_idxs, size=max_masks)
        mask_idxs = np.full_like(mask_idxs, False)
        mask_idxs[true_idxs] = True

    buffer = io.StringIO()
    mask_idx = 0
    for i, s in enumerate(splits):
        buffer.write(s)
        if i < len(splits) - 1:
            buffer.write(" ")
            if mask_idxs[i]:
                buffer.write(func(mask_idx))
                buffer.write(" ")
                mask_idx += 1

    return buffer.getvalue()
