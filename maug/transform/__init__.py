"""
This package provides transforms to apply to a given dataset. Each
transform receives a set of records and returns a new synthetic dataset with
the generated records.
"""
from maug.transform.base import Transform

from maug.transform.deletion import RandomDelete, SpanDelete, PunctSpanDelete

from maug.transform.mask_and_fill import MaskAndFill

from maug.transform.mistranslation import NamedEntityShuffle, Negation
