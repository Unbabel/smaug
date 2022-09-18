import functools
import re

from typing import Optional

from smaug import random
from smaug.mask import base
from smaug.mask import func
from smaug.model import MaskingPattern
from smaug.typing import Text


_NUM_REGEX = re.compile(r"[-+]?\.?(\d+[.,])*\d+")


class Number(base.Mask):
    """Masks numbers in a text according to a regex.

    Args:
        mask_token: token to replace detected numbers.
    """

    def __init__(
        self,
        pattern: MaskingPattern = None,
        max_mask: Optional[int] = None,
    ):
        super(Number, self).__init__(pattern)
        self.__max = max_mask
        self.__rng = random.numpy_seeded_rng()

    @functools.singledispatchmethod
    def __call__(self, text: Text) -> Text:
        raise NotImplementedError(f"Invalid type {type(text)}")

    @__call__.register
    def _(self, text: str):
        return self.__mask(text)

    @__call__.register
    def _(self, text: list):
        return [self.__mask(t) for t in text]

    def __mask(self, text: str):
        matches = _NUM_REGEX.finditer(text)
        if self.__max:
            matches = list(matches)
            if len(matches) > self.__max:
                matches = self.__rng.choice(matches, self.__max, replace=False)
        intervals = [m.span() for m in matches]
        return func.mask(text, intervals, self.pattern)
