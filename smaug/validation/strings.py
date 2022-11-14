import re
import collections

from nltk.metrics import edit_distance
from typing import Dict, List, Optional

from smaug import pipeline
from smaug.validation import base


class NotEqual(base.CmpBased):
    """Filters critical records that are equal to the generated ones."""

    def _verify(
        self,
        original: str,
        critical: str,
    ) -> bool:
        return original != critical


class MinRelativeLength(base.CmpBased):
    """Filters critical records that are to small compared to the original one.

    Args:
        threshold: minimum ratio len(critical) / len(original) that should be accepted.
        critical_field: Field inside the perturbations dictionary with the perturbation
            to test.
    """

    def __init__(
        self,
        threshold: float,
        critical_field: Optional[str] = None,
    ):
        super().__init__(critical_field=critical_field)
        self.__threshold = threshold

    def _verify(
        self,
        original: str,
        critical: str,
    ) -> bool:
        original_len = len(original)
        critical_len = len(critical)
        return critical_len / original_len >= self.__threshold


class NoRegexMatch(base.Validation):
    """Excludes critical sentences that match a given regex."""

    def __init__(
        self,
        pattern: str,
        critical_field: Optional[str] = None,
    ):
        super().__init__(critical_field=critical_field)
        self.__pattern = re.compile(pattern)

    def __call__(self, records: List[pipeline.State]) -> List[pipeline.State]:
        for r in records:
            if self.critical_field not in r.perturbations:
                continue
            if self.__pattern.search(r.perturbations[self.critical_field]) is not None:
                base.del_perturbation(self.critical_field, r)
        return records


class GeqEditDistance(base.CmpBased):
    """Filters perturbations with a small minimum edit distance to the original sentences.

    Args:
        min_dist: minimum edit distance that should be accepted.
        level: level at which to measure the minimum edit distance. Can be word or char.
        critical_field: Field inside the perturbations dictionary with the perturbation
            to test.
    """

    __LEVELS = ("char", "word")

    def __init__(
        self,
        min_dist: int,
        level: str = "char",
        critical_field: Optional[str] = None,
    ):
        super().__init__(critical_field=critical_field)
        self.__min_dist = min_dist
        if level not in self.__LEVELS:
            raise ValueError(f"Unknown level {level}: must be one of {self.__LEVELS}.")
        self.__level = level

    def _verify(
        self,
        original: str,
        critical: str,
    ) -> bool:
        if self.__level == "word":
            original = original.split()
            critical = critical.split()
        return edit_distance(original, critical) >= self.__min_dist


class LeqCharInsertions(base.CmpBased):
    """Filters perturbations with many insertions of specific caracters when compared to the original.

    This validation takes a set of characters and adds up how many insertions of these charactes
    the perturbed sentence has. If this number is over a threshold, the perturbation is rejected.

    Args:
        chars: String of characters to consider (each individual character will be considered).
        max_insertions: Maximum number of insertions.
        critical_field: Field inside the perturbations dictionary with the perturbation
            to test.
    """

    def __init__(
        self,
        chars: str,
        max_insertions: int = 0,
        critical_field: Optional[str] = None,
    ):
        super().__init__(critical_field=critical_field)
        self._chars = set(chars)
        self._max_ins = max_insertions

    def _verify(
        self,
        original: str,
        critical: str,
    ) -> bool:
        original_chars = (c for c in original if c in self._chars)
        critical_chars = (c for c in critical if c in self._chars)
        original_counts = collections.Counter(original_chars)
        critical_counts = collections.Counter(critical_chars)
        insertions = critical_counts - original_counts
        return sum(insertions.values()) <= self._max_ins
