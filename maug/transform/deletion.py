import abc
import io
import itertools
import numpy as np
import re

from typing import Dict, List, Optional

from maug import random
from maug.transform import base
from maug.transform import error
from maug._itertools import repeat_items


class Deletion(base.Transform, abc.ABC):
    """Base class for transforms that remove critical content in the translation.

    Args:
        name: name of the Transform.
        original_field: name of the field to transform in the received records.
        perturbations_field: Field to add to the original records to store
            the transformed sentences. This field is a dictionary with
            the transformation name as keys and the perturbed sentences as values.
        critical_field: Field to add inside the perturbations dictionary.
        num_samples: number of critical samples that should be generated for each
            original record."""

    def __init__(
        self,
        name: str,
        num_samples: int = 1,
        original_field: Optional[str] = None,
        perturbations_field: Optional[str] = None,
        critical_field: Optional[str] = None,
    ):
        super().__init__(
            name=name,
            original_field=original_field,
            perturbations_field=perturbations_field,
            critical_field=critical_field,
            error_type=error.ErrorType.DELETION,
        )
        self.__num_samples = num_samples

    def __call__(self, original: List[Dict]) -> List[Dict]:
        repeated_items = list(repeat_items(original, self.__num_samples))
        for orig in repeated_items:
            if self.perturbations_field not in orig:
                orig[self.perturbations_field] = {}
            perturbation = self._transform(orig[self.original_field])
            if perturbation:
                orig[self.perturbations_field][self.critical_field] = perturbation
        return repeated_items

    @abc.abstractmethod
    def _transform(self, sentence: str) -> Optional[str]:
        pass


class RandomDelete(Deletion):
    """Deletes random words from the translation.

    Args:
        p: probability of deleting a word
        original_field: name of the field to transform in the received records.
        perturbations_field: Field to add to the original records to store
            the transformed sentences. This field is a dictionary with
            the transformation name as keys and the perturbed sentences as values.
        critical_field: Field to add inside the perturbations dictionary.
        num_samples: number of critical samples to generate for each original
            record.
    """

    __NAME = "random-delete"

    def __init__(
        self,
        num_samples: int = 1,
        p: int = 0.2,
        original_field: Optional[str] = None,
        perturbations_field: Optional[str] = None,
        critical_field: Optional[str] = None,
    ):
        super(RandomDelete, self).__init__(
            name=self.__NAME,
            original_field=original_field,
            perturbations_field=perturbations_field,
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
        original_field: Optional[str] = None,
        perturbations_field: Optional[str] = None,
        critical_field: Optional[str] = None,
    ):
        super(SpanDelete, self).__init__(
            name=self.__NAME,
            num_samples=num_samples,
            original_field=original_field,
            perturbations_field=perturbations_field,
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


class PunctSpanDelete(Deletion):
    """Removes a span between two punctuation symbols.

    The considered symbols are .,!? .

    Args:
        low: minimum number of words for a span to be eligible for deletion.
        high: maximum number of words for a span to be eligible for deletion.
        original_field: name of the field to transform in the received records.
        perturbations_field: Field to add to the original records to store
            the transformed sentences. This field is a dictionary with
            the transformation name as keys and the perturbed sentences as values.
        critical_field: Field to add inside the perturbations dictionary.
        num_samples: number of critical samples that should be generated for each
            original record.
    """

    __NAME = "punct-span-delete"
    _PUNCT = re.compile(r"[.!?,]")

    def __init__(
        self,
        low: int = 4,
        high: int = 10,
        num_samples: int = 1,
        original_field: Optional[str] = None,
        perturbations_field: Optional[str] = None,
        critical_field: Optional[str] = None,
    ):
        super().__init__(
            name=self.__NAME,
            num_samples=num_samples,
            original_field=original_field,
            perturbations_field=perturbations_field,
            critical_field=critical_field,
        )
        self.__low = low
        self.__high = high
        self.__rng = random.numpy_seeded_rng()

    def _transform(self, sentence: str) -> Optional[str]:
        spans = self._PUNCT.split(sentence)
        # Indexes of spans that can be dropped.
        possible_drop_idxs = [
            i for i, s in enumerate(spans) if self.__low < len(s.split()) < self.__high
        ]
        if len(possible_drop_idxs) <= 2:
            return None

        idx_to_drop = self.__rng.choice(possible_drop_idxs)
        buffer = io.StringIO()
        line_idx = 0
        for i, span in enumerate(spans):
            # Skip span and punctuation after.
            if i == idx_to_drop:
                line_idx += len(span) + 1
                continue
            buffer.write(span)
            line_idx += len(span)
            if i < len(spans) - 1:
                buffer.write(sentence[line_idx])
                line_idx += 1
        return buffer.getvalue()
