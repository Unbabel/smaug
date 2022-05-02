import re
import typing


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
    """

    def __init__(
        self,
        threshold: float,
        original_field: typing.Optional[str] = None,
        critical_field: typing.Optional[str] = None,
    ):
        super().__init__(
            original_field=original_field,
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
        original_field: typing.Optional[str] = None,
        critical_field: typing.Optional[str] = None,
    ):
        super().__init__(
            original_field=original_field,
            critical_field=critical_field,
        )
        self.__pattern = re.compile(pattern)

    def __call__(self, records: typing.List[typing.Dict]) -> typing.List[typing.Dict]:
        for r in records:
            if (
                self.critical_field in r
                and self.__pattern.search(r[self.critical_field]) is not None
            ):
                del r[self.critical_field]
        return records
