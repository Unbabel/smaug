"""
This package offers validation utils to verify and filter generated records
that do not meet some specification
"""

from maug.validation.base import Validation, Sequential

from maug.validation.named_entities import EqualNamedEntityCount

from maug.validation.nli import IsContradiction

from maug.validation.numerical import EqualNumbersCount

from maug.validation.strings import (
    GeqEditDistance,
    MinRelativeLength,
    NotEqual,
    NoRegexMatch,
)
