import abc

from smaug.model.typing import MaskingPattern
from smaug.typing import Text
from smaug._itertools import ResetableIterator


class Mask(abc.ABC):
    """Base class for mask objects that apply masks to input sentences.

    Attributes:
        pattern: Masking pattern that will be replaced in the sentences
    """

    def __init__(self, pattern: MaskingPattern = None):
        self.__pattern = pattern

    @property
    def pattern(self):
        if not self.__pattern:
            raise ValueError("pattern not defined.")
        return self.__pattern

    @pattern.setter
    def pattern(self, value):
        if self.__pattern:
            raise ValueError("pattern already defined")
        self.__pattern = value

    @abc.abstractmethod
    def __call__(self, text: Text) -> Text:
        pass


class MaskIterator(ResetableIterator):
    """Wraps a masking pattern, returning always the next mask to use."""

    def __init__(self, pattern: MaskingPattern) -> None:
        if isinstance(pattern, str):
            next_fn = lambda: pattern
            reset_fn = lambda: None
        elif isinstance(pattern, ResetableIterator):
            next_fn = pattern.__next__
            reset_fn = pattern.reset
        else:
            raise ValueError(f"Unknown pattern type: {type(pattern)}")

        self._next_fn = next_fn
        self._reset_fn = reset_fn

    def __next__(self):
        return self._next_fn()

    def reset(self):
        self._reset_fn()
