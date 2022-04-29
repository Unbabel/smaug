"""
This package offers an interface for several models so that they can be
interchangeably used throughout this project.
"""

from maug.model.base import MaskedLanguageModel, Text2Text, TokenClassification
from maug.model.mt5 import MT5
from maug.model.m2m100 import M2M100
from maug.model.opus_mt import OpusMT
from maug.model.polyjuice import NegPolyjuice
from maug.model.roberta import RobertaMNLI
from maug.model.stanza_ner import StanzaNER
from maug.model.typing import MaskingPattern
