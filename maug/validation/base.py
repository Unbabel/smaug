import abc
import typing

_DEFAULT_ORIGINAL_FIELD = "original"
_DEFAULT_CRITICAL_FIELD = "critical"


class Validation(abc.ABC):
    """Filters critical records that do not meet some specification."""

    _original_field: str
    _critical_field: str

    def __init__(
        self,
        original_field: typing.Optional[str] = None,
        critical_field: typing.Optional[str] = None,
    ) -> None:
        if original_field is None:
            original_field = _DEFAULT_ORIGINAL_FIELD
        if critical_field is None:
            critical_field = _DEFAULT_CRITICAL_FIELD
        self._original_field = original_field
        self._critical_field = critical_field

    @property
    def original_field(self):
        return self._original_field

    @property
    def critical_field(self):
        return self._critical_field

    @abc.abstractmethod
    def __call__(self, records: typing.List[typing.Dict]) -> typing.List[typing.Dict]:
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

    def __call__(self, records: typing.List[typing.Dict]) -> typing.List[typing.Dict]:
        for val in self.__vals:
            records = val(records)
        return records


class CmpBased(Validation, abc.ABC):
    """Filters critical records based on comparing records."""

    def __call__(self, records: typing.List[typing.Dict]) -> typing.List[typing.Dict]:
        for r in records:
            if self.critical_field in r and not self._verify(
                r[self.original_field], r[self.critical_field]
            ):
                del r[self.critical_field]
        return records

    @abc.abstractmethod
    def _verify(self, original: str, critical: str) -> bool:
        pass
