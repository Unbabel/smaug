import abc

from typing import List, Dict, Optional

_DEFAULT_ORIGINAL_FIELD = "original"
_DEFAULT_PERTURBATIONS_FIELD = "perturbations"
_DEFAULT_CRITICAL_FIELD = "critical"


class Validation(abc.ABC):
    """Filters critical records that do not meet some specification.

    Args:
        original_field: Field in the original records to transform.
        perturbations_field: Field with the perturbations added by the transforms.
            This field is a dictionary with the transform name as keys and the
            perturbed sentences as values.
        critical_field: Field inside the perturbations dictionary with the perturbation
            to test.
    """

    _original_field: str
    _perturbations_field: str
    _critical_field: str

    def __init__(
        self,
        original_field: Optional[str] = None,
        perturbations_field: Optional[str] = None,
        critical_field: Optional[str] = None,
    ) -> None:
        if original_field is None:
            original_field = _DEFAULT_ORIGINAL_FIELD
        if perturbations_field is None:
            perturbations_field = _DEFAULT_PERTURBATIONS_FIELD
        if critical_field is None:
            critical_field = _DEFAULT_CRITICAL_FIELD
        self._original_field = original_field
        self._perturbations_field = perturbations_field
        self._critical_field = critical_field

    @property
    def original_field(self):
        return self._original_field

    @property
    def perturbations_field(self):
        return self._perturbations_field

    @property
    def critical_field(self):
        return self._critical_field

    @abc.abstractmethod
    def __call__(self, records: List[Dict]) -> List[Dict]:
        """Filters critical records that do not meet some specification.

        Args:
            records: dicts with original and critical sentences to validate.

        Returns:
            dicts that passed the verifications.
        """
        pass


class Sequential(Validation):
    """Applies multiple validations sequentially."""

    def __init__(self, *args: Validation) -> None:
        super().__init__()
        self.__vals = list(args)

    def __call__(self, records: List[Dict]) -> List[Dict]:
        for val in self.__vals:
            records = val(records)
        return records


class CmpBased(Validation, abc.ABC):
    """Filters critical records based on comparing records."""

    def __call__(self, records: List[Dict]) -> List[Dict]:
        for r in records:
            if self.perturbations_field not in r:
                continue
            perturbations = r[self.perturbations_field]
            if self.critical_field not in perturbations:
                continue
            if not self._verify(
                r[self.original_field], perturbations[self.critical_field]
            ):
                del perturbations[self.critical_field]
        return records

    @abc.abstractmethod
    def _verify(self, original: str, critical: str) -> bool:
        pass
