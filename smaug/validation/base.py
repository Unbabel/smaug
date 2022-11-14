import abc

from typing import List, Optional

from smaug import pipeline

_DEFAULT_CRITICAL_FIELD = "critical"


class Validation(abc.ABC):
    """Filters critical records that do not meet some specification.

    Args:
        critical_field: Field inside the perturbations dictionary with the perturbation
            to test.
    """

    _critical_field: str

    def __init__(
        self,
        critical_field: Optional[str] = None,
    ) -> None:
        if critical_field is None:
            critical_field = _DEFAULT_CRITICAL_FIELD
        self._critical_field = critical_field

    @property
    def critical_field(self):
        return self._critical_field

    @abc.abstractmethod
    def __call__(self, records: List[pipeline.State]) -> List[pipeline.State]:
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

    def __call__(self, records: List[pipeline.State]) -> List[pipeline.State]:
        for val in self.__vals:
            records = val(records)
        return records


class CmpBased(Validation, abc.ABC):
    """Filters critical records based on comparing records."""

    def __call__(self, records: List[pipeline.State]) -> List[pipeline.State]:
        for r in records:
            if self.critical_field not in r.perturbations:
                continue
            if not self._verify(r.original, r.perturbations[self.critical_field]):
                del_perturbation(self.critical_field, r)
        return records

    @abc.abstractmethod
    def _verify(self, original: str, critical: str) -> bool:
        pass


def del_perturbation(field: str, state: pipeline.State):
    if field in state.perturbations:
        del state.perturbations[field]
    if field in state.metadata:
        del state.metadata[field]
