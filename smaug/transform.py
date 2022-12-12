import abc
import enum
import io
import itertools
import numpy as np
import re

from typing import Callable, Dict, Iterable, List, Optional, Set

from smaug import core
from smaug import pipeline
from smaug import _itertools
from smaug.ops import ner


class ErrorType(enum.Enum):
    """Defines error types that a specific transform produces."""

    UNDEFINED = 0
    """The error type is not defined."""

    NOT_CRITICAL = 1
    """The transform does not induce any critical error in the translation. 
    Nevertheless, if induced on a critical error example, the example should 
    still be classified as critical."""

    MISTRANSLATION = 2
    """The transform creates a mistranslation error. The content of the 
    translation has a different meaning when compared to the source, is not 
    translated (remains in the source language), or is translated into 
    gibberish."""

    HALLUCINATION = 3
    """The translation creates an hallucination, where new content is added to
    the translation."""

    DELETION = 4
    """The translation creates a deletion error, where critical content is
    removed from the translation."""


_DEFAULT_CRITICAL_FIELD = "critical"


class Transform(abc.ABC):
    """Base class for all transforms.

    Attributes:
        name: Name of the transform.
        critical_field: Field to add inside the perturbations dictionary.
        error_type: Error type that the transform induces in the synthetic
            dataset.
    """

    _name: str
    _critical_field: str
    _error_type: ErrorType

    def __init__(
        self,
        name: str,
        critical_field: Optional[str] = None,
        error_type: ErrorType = ErrorType.UNDEFINED,
    ):
        if critical_field is None:
            critical_field = _DEFAULT_CRITICAL_FIELD

        self._name = name
        self._critical_field = critical_field
        self._error_type = error_type

    @property
    def name(self):
        return self._name

    @property
    def critical_field(self):
        return self._critical_field

    @property
    def error_type(self):
        return self._error_type

    @abc.abstractmethod
    def __call__(self, original: List[pipeline.State]) -> List[pipeline.State]:
        """Transforms non-critical batch into a critical batch.

        Args:
            original: The data to be transformed.

        Returns:
            generated critical records. The transform returns a list
            of dicts with the original data and the generated perturbations
            inside a dictionary. The perturbations dictionary is indexed
            by transform name and has the perturbed sentences as values.
        """
        pass


class Deletion(Transform, abc.ABC):
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
            error_type=ErrorType.DELETION,
        )
        self.__num_samples = num_samples

    def __call__(self, original: List[pipeline.State]) -> List[pipeline.State]:
        repeated_items: List[pipeline.State] = list(
            _itertools.repeat_items(original, self.__num_samples)
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
        rng: np.random.Generator,
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
        self.__rng = rng

    def _transform(self, sentence: str) -> str:
        splits = sentence.split()
        return " ".join(filter(lambda _: self.__rng.random() < self.__p, splits))


class SpanDelete(Deletion):

    __NAME = "span-delete"

    def __init__(
        self,
        rng: np.random.Generator,
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
        self.__rng = rng

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
        rng: np.random.Generator,
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
        self.__rng = rng

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


class Mistranslation(Transform, abc.ABC):
    """Base class for transforms that mistranslate critical content in the
    translation.
    """

    def __init__(
        self,
        name: str,
        critical_field: Optional[str] = None,
    ):
        super().__init__(
            name=name,
            critical_field=critical_field,
            error_type=ErrorType.MISTRANSLATION,
        )


class NamedEntityShuffle(Mistranslation):
    """Shuffles the named entities in a sentence.

    This transform uses the smaug.model.StanzaNER model to identify entities
    according to several tags and then shuffles entities with the same tag.

    Args:
        entities: Entity tags to consider. They should be a subset of the tags
            returned by the StanzaNER model. If not specified, all tags are
            used.
        side: Sentence where to apply the shuffling. Must be one of
            { source, target }, for the source and target sentences.
        lang: Language of the sentences where the transform is applied.
        critical_field: Field to add inside the perturbations dictionary.
    """

    __ner: Callable
    __entities: Set[str]

    __NAME = "named-entity-shuffle"

    __FOUR_TAGS = ("PER", "LOC", "ORG", "MISC")

    __EIGHTEEN_TAGS = (
        "PERSON",
        "NORP",  # Nationalities / Religious / Political Group
        "FAC",  # Facility
        "ORG",  # Organization
        "GPE",  # Countries / Cities / States
        "LOC",  # Location
        "PRODUCT",
        "EVENT",
        "WORK_OF_ART",
        "LAW",
        "LANGUAGE",
    )

    __BLNSP_TAGS = ("EVENT", "LOCATION", "ORGANIZATION", "PERSON", "PRODUCT")

    __ITALIAN_TAGS = ("LOC", "ORG", "PER")

    __MYANMAR_TAGS = (
        "LOC",
        "NE",  # Miscellaneous
        "ORG",
        "PNAME",
        "RACE",
    )

    __DEFAULT_TAGS: Dict[str, Iterable[str]] = {
        "af": __FOUR_TAGS,
        "ar": __FOUR_TAGS,
        "bg": __BLNSP_TAGS,
        "zh": __EIGHTEEN_TAGS,
        "nl": __FOUR_TAGS,
        "en": __EIGHTEEN_TAGS,
        "fi": __FOUR_TAGS,
        "fr": __FOUR_TAGS,
        "de": __FOUR_TAGS,
        "hu": __FOUR_TAGS,
        "it": __ITALIAN_TAGS,
        "my": __MYANMAR_TAGS,
        "ru": __FOUR_TAGS,
        "es": __FOUR_TAGS,
        "uk": __FOUR_TAGS,
        "vi": __FOUR_TAGS,
    }

    def __init__(
        self,
        lang: str,
        ner_func: Callable[[core.DataLike[str]], core.Data],
        rng: np.random.Generator,
        entities: Optional[Iterable[str]] = None,
        critical_field: Optional[str] = None,
    ):
        super().__init__(name=self.__NAME, critical_field=critical_field)

        if entities is None:
            entities = ner.stanza_ner_tags(lang)
        entities = set(entities)
        for e in entities:
            if e not in ner.stanza_ner_tags(lang):
                raise ValueError(f"Unknown entity type: {e}")

        self.__ner = ner_func
        self.__entities = entities
        self.__rng = rng

    def __call__(self, original: List[pipeline.State]) -> List[pipeline.State]:
        for orig in original:
            perturbed = self.__shuffle_all_entity_types(orig.original)
            if perturbed is not None:
                orig.perturbations[self.critical_field] = perturbed

        return original

    def __shuffle_all_entity_types(self, sentence: str) -> Optional[str]:
        current = sentence

        for entity in self.__entities:
            # Continue if more than two entities were detected in total.
            current, should_continue = self.__shuffle_single_entity_type(
                current,
                entity,
            )
            if not should_continue:
                break

        return current

    def __shuffle_single_entity_type(self, sentence, entity):
        entities = self.__ner(sentence).item().entities
        # If less than two entities were detected all subsequent calls will
        # not lead to shuffling, and we should stop trying.
        if len(entities) < 2:
            return sentence, False
        intervals = [(e.start_char, e.end_char) for e in entities if e.type == entity]
        # Can only be applied for at least two entities
        if len(intervals) < 2:
            return sentence, True

        ordered_indexes = np.arange(len(intervals))
        swapped_indexes = ordered_indexes.copy()
        self.__rng.shuffle(swapped_indexes)
        while np.all(np.equal(ordered_indexes, swapped_indexes)):
            self.__rng.shuffle(swapped_indexes)

        # Build the final string by appending a non swapped chunks and then the
        # swapped named entity.
        # In the final index, also append the non swapped chunk.
        buffer = io.StringIO()
        for ordered_idx, swapped_idx in enumerate(swapped_indexes):
            # Get non swapped chunk
            if ordered_idx == 0:
                non_swapped = sentence[: intervals[ordered_idx][0]]
            else:
                non_swap_start = intervals[ordered_idx - 1][1]
                non_swap_end = intervals[ordered_idx][0]
                non_swapped = sentence[non_swap_start:non_swap_end]
            buffer.write(non_swapped)

            # Get swapped chunk
            interval_slice = slice(*intervals[swapped_idx])
            buffer.write(sentence[interval_slice])

            # In the last index, write the final non swapped chunk which is the
            # remaining sentence
            if ordered_idx == len(intervals) - 1:
                buffer.write(sentence[intervals[ordered_idx][1] :])

        return buffer.getvalue(), True


class Negation(Mistranslation):
    """Negates the original sentence.

    Not all sentences can be negated, and so not all records will be returned.

    Args:
        neg_polyjuice: Polyjuice model conditioned on negation.
        num_samples: Number of critical records to generate for each original records.
        critical_field: Field to add inside the perturbations dictionary.
    """

    __NAME = "negation"

    def __init__(
        self,
        neg_polyjuice,
        num_samples: int = 1,
        critical_field: Optional[str] = None,
    ):
        super().__init__(
            self.__NAME,
            critical_field=critical_field,
        )
        self.__neg_polyjuice = neg_polyjuice
        self.__num_samples = num_samples

    def __call__(self, original: List[pipeline.State]) -> List[pipeline.State]:
        repeated_items: List[pipeline.State] = list(
            _itertools.repeat_items(original, self.__num_samples)
        )

        original_sentences = [x.original for x in repeated_items]
        negated = self.__neg_polyjuice(original_sentences)

        for orig, n in zip(repeated_items, negated):
            if n is None:
                continue
            orig.perturbations[self.critical_field] = n

        return repeated_items


class MaskAndFill(Transform):
    """Generates critical errors by masking and filling sentences.

    This class generates critical errors based on the following steps:

    Perturbing either the source or the target sentence by masking it and then
    filling the masked tokens.

    The perturbed sentence is verified to be different from the original one.

    Args:
        mask: Mask object to use when masking sentences.
        fill: Model to fill the masked sentences.
        num_samples: Number of generated samples to create.
        critical_field: Field to add inside the perturbations dictionary.
    """

    __NAME = "mask-and-fill"

    def __init__(
        self,
        mask,
        fill,
        num_samples: int = 1,
        critical_field: Optional[str] = None,
    ):
        super().__init__(
            name=self.__NAME,
            error_type=ErrorType.UNDEFINED,
            critical_field=critical_field,
        )
        self.__masking = mask
        self.__fill = fill
        self.__num_samples = num_samples

    def __call__(self, original: List[pipeline.State]) -> List[pipeline.State]:
        repeated_items: List[pipeline.State] = list(
            _itertools.repeat_items(original, self.__num_samples)
        )

        original_sentences = [x.original for x in repeated_items]
        masked = self.__masking(original_sentences)
        filled = self.__fill(masked)

        for orig, t in zip(repeated_items, filled.text):
            orig.perturbations[self.critical_field] = t

        for orig, s in zip(repeated_items, filled.spans):
            orig.metadata[self.critical_field] = s

        return repeated_items
