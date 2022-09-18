import functools
import io
import numpy as np

from typing import Optional

from smaug import random
from smaug.mask import base
from smaug.mask import func
from smaug.typing import Text


class RandomReplace(base.Mask):
    """Masks random words in an input according to a probability."""

    def __init__(self, pattern=None, p=0.1):
        super().__init__(pattern)
        self.__p = (1 - p, p)
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
        splits = text.split()
        mask = self.__rng.choice([False, True], size=len(splits), p=self.__p)
        mask_iter = func.MaskIterator(self.pattern)
        mask_iter.reset()
        splits = [next(mask_iter) if m else s for s, m in zip(splits, mask)]
        return " ".join(splits)


class RandomInsert(base.Mask):
    """Randomly inserts masks between words in an input according to a
    probability.
    """

    def __init__(self, pattern=None, p: float = 0.2, max_masks: Optional[int] = None):
        super(RandomInsert, self).__init__(pattern)
        self.__p = p
        self.__max_masks = max_masks
        self.__rng = random.numpy_seeded_rng()

    @functools.singledispatchmethod
    def __call__(self, text: Text) -> Text:
        raise NotImplementedError(f"__call__ not implemented for type {type(text)}")

    @__call__.register
    def _(self, text: str) -> str:
        return self.__process_single(text)

    @__call__.register
    def _(self, text: list) -> list:
        return [self.__process_single(t) for t in text]

    def __process_single(self, text: str) -> str:
        splits = text.split()
        # Not enought splits to insert mask.
        if len(splits) < 2:
            return text

        mask_idxs = self.__rng.choice(
            [False, True], size=len(splits) - 1, p=[self.__p, 1 - self.__p]
        )
        (true_idxs,) = np.nonzero(mask_idxs)
        if self.__max_masks is not None and len(true_idxs) > self.__max_masks:
            true_idxs = self.__rng.choice(true_idxs, size=self.__max_masks)
            mask_idxs = np.full_like(mask_idxs, False)
            mask_idxs[true_idxs] = True

        buffer = io.StringIO()
        mask_iter = base.MaskIterator(self.pattern)
        mask_iter.reset()
        for i, s in enumerate(splits):
            buffer.write(s)
            if i < len(splits) - 1:
                buffer.write(" ")
                if mask_idxs[i]:
                    buffer.write(next(mask_iter))
                    buffer.write(" ")

        return buffer.getvalue()
