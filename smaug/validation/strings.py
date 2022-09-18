import re
import collections

from nltk.metrics import edit_distance
from typing import Dict, Iterable, List, Optional

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
        original_field: Field in the original records to transform.
        perturbations_field: Field with the perturbations added by the transforms.
            This field is a dictionary with the transform name as keys and the
            perturbed sentences as values.
        critical_field: Field inside the perturbations dictionary with the perturbation
            to test.
    """

    def __init__(
        self,
        threshold: float,
        original_field: Optional[str] = None,
        perturbations_field: Optional[str] = None,
        critical_field: Optional[str] = None,
    ):
        super().__init__(
            original_field=original_field,
            perturbations_field=perturbations_field,
            critical_field=critical_field,
        )
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
        original_field: Optional[str] = None,
        perturbations_field: Optional[str] = None,
        critical_field: Optional[str] = None,
    ):
        super().__init__(
            original_field=original_field,
            perturbations_field=perturbations_field,
            critical_field=critical_field,
        )
        self.__pattern = re.compile(pattern)

    def __call__(self, records: List[Dict]) -> List[Dict]:
        for r in records:
            if self.perturbations_field not in r:
                continue
            perturbations = r[self.perturbations_field]
            if self.critical_field not in perturbations:
                continue
            if self.__pattern.search(perturbations[self.critical_field]) is not None:
                del perturbations[self.critical_field]
        return records


class GeqEditDistance(base.CmpBased):
    """Filters perturbations with a small minimum edit distance to the original sentences.

    Args:
        min_dist: minimum edit distance that should be accepted.
        level: level at which to measure the minimum edit distance. Can be word or char.
        original_field: Field in the original records to transform.
        perturbations_field: Field with the perturbations added by the transforms.
            This field is a dictionary with the transform name as keys and the
            perturbed sentences as values.
        critical_field: Field inside the perturbations dictionary with the perturbation
            to test.
    """

    __LEVELS = ("char", "word")

    def __init__(
        self,
        min_dist: int,
        level: str = "char",
        original_field: Optional[str] = None,
        perturbations_field: Optional[str] = None,
        critical_field: Optional[str] = None,
    ):
        super().__init__(
            original_field=original_field,
            perturbations_field=perturbations_field,
            critical_field=critical_field,
        )
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
        original_field: Field in the original records to transform.
        perturbations_field: Field with the perturbations added by the transforms.
            This field is a dictionary with the transform name as keys and the
            perturbed sentences as values.
        critical_field: Field inside the perturbations dictionary with the perturbation
            to test.
    """

    def __init__(
        self,
        chars: str,
        max_insertions: Optional[int] = 0,
        original_field: Optional[str] = None,
        perturbations_field: Optional[str] = None,
        critical_field: Optional[str] = None,
    ):
        super().__init__(
            original_field=original_field,
            perturbations_field=perturbations_field,
            critical_field=critical_field,
        )
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
