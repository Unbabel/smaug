"""
This package offers several masking utilities following different masking
patterns.
"""

from smaug.mask.base import Mask, MaskIterator

from smaug.mask.named_entity import NamedEntity

from smaug.mask.numerical import Number

from smaug.mask.random import RandomInsert, RandomReplace
