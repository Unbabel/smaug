import re

from nltk.metrics import edit_distance
from typing import Dict, List, Optional

from maug.validation import base


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
    """Filters critical records with a small edit distance to the original one.

    Args:
        min_dist: minimum edit distance that should be accepted.
        original_field: Field in the original records to transform.
        perturbations_field: Field with the perturbations added by the transforms.
            This field is a dictionary with the transform name as keys and the
            perturbed sentences as values.
        critical_field: Field inside the perturbations dictionary with the perturbation
            to test.
    """

    def __init__(
        self,
        min_dist: int,
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

    def _verify(
        self,
        original: str,
        critical: str,
    ) -> bool:
        return edit_distance(original, critical) >= self.__min_dist
