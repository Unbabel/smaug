"""
This package offers several masking utilities following different masking
patterns.
"""

from maug.mask.base import Mask, MaskIterator

from maug.mask.named_entity import NamedEntity

from maug.mask.numerical import Number

from maug.mask.random import RandomInsert, RandomReplace
