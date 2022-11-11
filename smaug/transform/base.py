import abc

from typing import List, Optional

from smaug import pipeline
from smaug.transform import error


_DEFAULT_CRITICAL_FIELD = "critical"


class Transform(abc.ABC):
    """Base class for all transforms.

    Attributes:
        name: Name of the transform.
        critical_field: Field to add inside the perturbations dictionary.
        error_type: Error type that the transform induces in the synthetic
            dataset.
    """

    _name: str
    _critical_field: str
    _error_type: error.ErrorType

    def __init__(
        self,
        name: str,
        critical_field: Optional[str] = None,
        error_type: error.ErrorType = error.ErrorType.UNDEFINED,
    ):
        if critical_field is None:
            critical_field = _DEFAULT_CRITICAL_FIELD

        self._name = name
        self._critical_field = critical_field
        self._error_type = error_type

    @property
    def name(self):
        return self._name

    @property
    def critical_field(self):
        return self._critical_field

    @property
    def error_type(self):
        return self._error_type

    @abc.abstractmethod
    def __call__(self, original: List[pipeline.State]) -> List[pipeline.State]:
        """Transforms non-critical batch into a critical batch.

        Args:
            original: The data to be transformed.

        Returns:
            generated critical records. The transform returns a list
            of dicts with the original data and the generated perturbations
            inside a dictionary. The perturbations dictionary is indexed
            by transform name and has the perturbed sentences as values.
        """
        pass
