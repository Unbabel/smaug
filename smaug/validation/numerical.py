import re

from smaug.validation import base


_NUM_REGEX = re.compile(r"[-+]?\.?(\d+[.,])*\d+")


class EqualNumbersCount(base.CmpBased):
    """Filters critical records that do NOT have the same numbers count."""

    def _verify(self, original: str, critical: str) -> bool:
        orig_count = len(_NUM_REGEX.findall(original))
        crit_count = len(_NUM_REGEX.findall(critical))
        return orig_count == crit_count
