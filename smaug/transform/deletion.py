import abc
import io
import itertools
import numpy as np
import re

from typing import List, Optional

from smaug import pipeline
from smaug import random
from smaug.transform import base
from smaug.transform import error
from smaug._itertools import repeat_items


class Deletion(base.Transform, abc.ABC):
    """Base class for transforms that remove critical content in the translation.

    Args:
        name: name of the Transform.
        critical_field: Field to add inside the perturbations dictionary.
        num_samples: number of critical samples that should be generated for each
            original record."""

    def __init__(
        self,
        name: str,
        num_samples: int = 1,
        critical_field: Optional[str] = None,
    ):
        super().__init__(
            name=name,
            critical_field=critical_field,
            error_type=error.ErrorType.DELETION,
        )
        self.__num_samples = num_samples

    def __call__(self, original: List[pipeline.State]) -> List[pipeline.State]:
        repeated_items: List[pipeline.State] = list(
            repeat_items(original, self.__num_samples)
        )
        for orig in repeated_items:
            perturbation = self._transform(orig.original)
            if perturbation:
                orig.perturbations[self.critical_field] = perturbation
        return repeated_items

    @abc.abstractmethod
    def _transform(self, sentence: str) -> Optional[str]:
        pass


class RandomDelete(Deletion):
    """Deletes random words from the translation.

    Args:
        p: probability of deleting a word
        critical_field: Field to add inside the perturbations dictionary.
        num_samples: number of critical samples to generate for each original
            record.
    """

    __NAME = "random-delete"

    def __init__(
        self,
        num_samples: int = 1,
        p: float = 0.2,
        critical_field: Optional[str] = None,
    ):
        super(RandomDelete, self).__init__(
            name=self.__NAME,
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
        critical_field: Optional[str] = None,
    ):
        super(SpanDelete, self).__init__(
            name=self.__NAME,
            num_samples=num_samples,
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

    Args:
        punct: punctuation symbols to consider.
        low: minimum number of words for a span to be eligible for deletion.
        high: maximum number of words for a span to be eligible for deletion.
        critical_field: Field to add inside the perturbations dictionary.
        num_samples: number of critical samples that should be generated for each
            original record.
    """

    __NAME = "punct-span-delete"

    def __init__(
        self,
        punct: str = ".,!?",
        low: int = 4,
        high: int = 10,
        num_samples: int = 1,
        critical_field: Optional[str] = None,
    ):
        super().__init__(
            name=self.__NAME,
            num_samples=num_samples,
            critical_field=critical_field,
        )
        self.__punct = re.compile(f"[{punct}]+")
        self.__low = low
        self.__high = high
        self.__rng = random.numpy_seeded_rng()

    def _transform(self, sentence: str) -> Optional[str]:
        spans = self.__punct.split(sentence)
        # Indexes of spans that can be dropped.
        # The first index is not considered as models are
        # more likelly to fail on the end of the sentence.
        possible_drop_idxs = [
            i
            for i, s in enumerate(spans)
            if i > 0 and self.__low < len(s.split()) < self.__high
        ]
        # Only delete when there are several subsentences,
        # to avoid deleting the entire content, making the
        # example trivial to identify.
        if len(possible_drop_idxs) < 2:
            return None

        idx_to_drop = self.__rng.choice(possible_drop_idxs)
        buffer = io.StringIO()
        sentence_idx = 0

        for i, span in enumerate(spans):
            if i != idx_to_drop:
                buffer.write(span)
            sentence_idx += len(span)

            if i < len(spans) - 1:
                punct_after_span = self.__punct.match(sentence, pos=sentence_idx)
                len_punct_after = punct_after_span.end() - punct_after_span.start()
                if i != idx_to_drop:
                    buffer.write(
                        sentence[sentence_idx : sentence_idx + len_punct_after]
                    )
                sentence_idx += len_punct_after

        sentence_no_span = buffer.getvalue().strip()
        # Too increase credibility of generated sentence,
        # replace last "," with "." .
        if not sentence_no_span.endswith((".", "?", "!")):
            sentence_no_span = f"{sentence_no_span[:-1]}."

        return sentence_no_span
