"""
This package offers validation utils to verify and filter generated records
that do not meet some specification
"""

from smaug.validation.base import Validation, Sequential

from smaug.validation.named_entities import EqualNamedEntityCount

from smaug.validation.nli import IsContradiction

from smaug.validation.numerical import EqualNumbersCount

from smaug.validation.strings import (
    GeqEditDistance,
    MinRelativeLength,
    NotEqual,
    NoRegexMatch,
    LeqCharInsertions,
)
