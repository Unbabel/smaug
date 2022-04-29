import abc
import itertools
import numpy as np
import typing

from maug import random
from maug.transform import base
from maug.transform import error
from maug._itertools import repeat_items


class Deletion(base.Transform, abc.ABC):
    """Base class for transforms that remove critical content in the translation.

    Args:
        name: name of the Transform.
        original_field: name of the field to transform in the received records.
        critical_field: name of the field with the critical sentence in the
            generated records.
        num_samples: number of critical samples that should be generated for each
            original record."""

    def __init__(
        self,
        name: str,
        num_samples: int = 1,
        original_field: typing.Optional[str] = None,
        critical_field: typing.Optional[str] = None,
    ):
        super().__init__(
            name=name,
            original_field=original_field,
            critical_field=critical_field,
            error_type=error.ErrorType.DELETION,
        )
        self.__num_samples = num_samples

    def __call__(self, original: typing.List[typing.Dict]) -> typing.List[typing.Dict]:
        repeated_items = repeat_items(original, self.__num_samples)
        return [
            {self.critical_field: self._transform(x[self.original_field]), **x}
            for x in repeated_items
        ]

    @abc.abstractmethod
    def _transform(self, sentence: str) -> str:
        pass


class RandomDelete(Deletion):
    """Deletes random words from the translation.

    Args:
        p: probability of deleting a word
        original_field: name of the field to transform in the received records.
        critical_field: name of the field with the critical sentence in the
            generated records.
        num_samples: number of critical samples to generate for each original
            record.
    """

    __NAME = "random-delete"

    def __init__(
        self,
        num_samples: int = 1,
        p: int = 0.2,
        original_field: typing.Optional[str] = None,
        critical_field: typing.Optional[str] = None,
    ):
        super(RandomDelete, self).__init__(
            name=self.__NAME,
            original_field=original_field,
            critical_field=critical_field,
            num_samples=num_samples,
        )
        self.__p = 1 - p
        self.__rng = random.numpy_seeded_rng()

    def _transform(self, sentence: str) -> str:
        splits = sentence.split()
        return " ".join(filter(lambda _: self.__rng.random() < self.__p, splits))


class SpanDelete(Deletion):

    __NAME = "span-delete"

    def __init__(
        self,
        min_size: float = 0.25,
        num_samples: int = 1,
        original_field: typing.Optional[str] = None,
        critical_field: typing.Optional[str] = None,
    ):
        super(SpanDelete, self).__init__(
            name=self.__NAME,
            num_samples=num_samples,
            original_field=original_field,
            critical_field=critical_field,
        )
        self.__min_size = min_size
        self.__rng = random.numpy_seeded_rng()

    def _transform(self, sentence: str) -> str:
        splits = sentence.split()
        num_splits = len(splits)

        lower_idx, higher_idx = 0, 0
        span_size = higher_idx - lower_idx
        while span_size / num_splits <= self.__min_size:
            lower_idx, higher_idx = self.__rng.choice(
                np.arange(num_splits),
                size=2,
                replace=False,
            )

            if lower_idx > higher_idx:
                lower_idx, higher_idx = higher_idx, lower_idx
            span_size = higher_idx - lower_idx

        critical_splits = itertools.chain(
            splits[:lower_idx],
            splits[higher_idx:],
        )
        return " ".join(critical_splits)
