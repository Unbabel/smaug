"""
This package provides transforms to apply to a given dataset. Each
transform receives a set of records and returns a new synthetic dataset with
the generated records.
"""
from smaug.transform.base import Transform

from smaug.transform.deletion import RandomDelete, SpanDelete, PunctSpanDelete

from smaug.transform.mask_and_fill import MaskAndFill

from smaug.transform.mistranslation import NamedEntityShuffle, Negation
