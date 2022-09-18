import abc

from typing import Dict, List, Optional

from smaug.transform import error


_DEFAULT_ORIGINAL_FIELD = "original"
_DEFAULT_PERTURBATIONS_FIELD = "perturbations"
_DEFAULT_CRITICAL_FIELD = "critical"


class Transform(abc.ABC):
    """Base class for all transforms.

    Attributes:
        name: Name of the transform.
        original_field: Field in the original records to transform.
        perturbations_field: Field to add to the original records to store
            the transformed sentences. This field is a dictionary with
            the transformation name as keys and the perturbed sentences as values.
        critical_field: Field to add inside the perturbations dictionary.
        error_type: Error type that the transform induces in the synthetic
            dataset.
    """

    _name: str
    _original_field: str
    _perturbations_field: str
    _critical_field: str
    _error_type: error.ErrorType

    def __init__(
        self,
        name: str,
        original_field: Optional[str] = None,
        perturbations_field: Optional[str] = None,
        critical_field: Optional[str] = None,
        error_type: error.ErrorType = error.ErrorType.UNDEFINED,
    ):
        if original_field is None:
            original_field = _DEFAULT_ORIGINAL_FIELD
        if perturbations_field is None:
            perturbations_field = _DEFAULT_PERTURBATIONS_FIELD
        if critical_field is None:
            critical_field = _DEFAULT_CRITICAL_FIELD

        self._name = name
        self._original_field = original_field
        self._perturbations_field = perturbations_field
        self._critical_field = critical_field
        self._error_type = error_type

    @property
    def name(self):
        return self._name

    @property
    def original_field(self):
        return self._original_field

    @property
    def perturbations_field(self):
        return self._perturbations_field

    @property
    def critical_field(self):
        return self._critical_field

    @property
    def error_type(self):
        return self._error_type

    @abc.abstractmethod
    def __call__(self, original: List[Dict]) -> List[Dict]:
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
