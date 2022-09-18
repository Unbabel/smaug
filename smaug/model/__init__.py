"""
This package offers an interface for several models so that they can be
interchangeably used throughout this project.
"""

from smaug.model.base import MaskedLanguageModel, Text2Text, TokenClassification
from smaug.model.mt5 import MT5
from smaug.model.m2m100 import M2M100
from smaug.model.opus_mt import OpusMT
from smaug.model.polyjuice import NegPolyjuice
from smaug.model.roberta import RobertaMNLI
from smaug.model.stanza_ner import StanzaNER
from smaug.model.typing import MaskingPattern
