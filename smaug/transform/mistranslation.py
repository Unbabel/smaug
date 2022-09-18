import abc
import io
import numpy as np

from typing import Dict, Iterable, List, Optional, Set

from smaug import model
from smaug import random
from smaug.transform import base
from smaug.transform import error
from smaug._itertools import repeat_items


class Mistranslation(base.Transform, abc.ABC):
    """Base class for transforms that mistranslate critical content in the
    translation.
    """

    def __init__(
        self,
        name: str,
        original_field: Optional[str] = None,
        perturbations_field: Optional[str] = None,
        critical_field: Optional[str] = None,
    ):
        super().__init__(
            name=name,
            original_field=original_field,
            perturbations_field=perturbations_field,
            critical_field=critical_field,
            error_type=error.ErrorType.MISTRANSLATION,
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
        original_field: name of the field to transform in the received records.
        perturbations_field: Field to add to the original records to store
            the transformed sentences. This field is a dictionary with
            the transformation name as keys and the perturbed sentences as values.
        critical_field: Field to add inside the perturbations dictionary.
    """

    __ner: model.StanzaNER
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

    __DEFAULT_TAGS = {
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
        model: model.StanzaNER,
        entities: Optional[Iterable[str]] = None,
        original_field: Optional[str] = None,
        perturbations_field: Optional[str] = None,
        critical_field: Optional[str] = None,
    ):
        super().__init__(
            name=self.__NAME,
            original_field=original_field,
            perturbations_field=perturbations_field,
            critical_field=critical_field,
        )

        if entities is None:
            entities = self.__DEFAULT_TAGS[lang]
        entities = set(entities)
        for e in entities:
            if e not in model.tags:
                raise ValueError(f"Unknown entity type: {e}")

        self.__ner = model
        self.__entities = entities
        self.__rng = random.numpy_seeded_rng()

    def __call__(self, original: List[Dict]) -> List[Dict]:
        for orig in original:
            if self.perturbations_field not in orig:
                orig[self.perturbations_field] = {}
            orig[self.perturbations_field][
                self.critical_field
            ] = self.__shuffle_all_entity_types(orig[self.original_field])

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
        entities = self.__ner(sentence).entities
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
        original_field: name of the field to transform in the received records.
        perturbations_field: Field to add to the original records to store
            the transformed sentences. This field is a dictionary with
            the transformation name as keys and the perturbed sentences as values.
        critical_field: Field to add inside the perturbations dictionary.
    """

    __NAME = "negation"

    def __init__(
        self,
        neg_polyjuice: model.NegPolyjuice,
        num_samples: int = 1,
        original_field: Optional[str] = None,
        perturbations_field: Optional[str] = None,
        critical_field: Optional[str] = None,
    ):
        super().__init__(
            self.__NAME,
            original_field=original_field,
            perturbations_field=perturbations_field,
            critical_field=critical_field,
        )
        self.__neg_polyjuice = neg_polyjuice
        self.__num_samples = num_samples

    def __call__(self, original: List[Dict]) -> List[Dict]:
        repeated_items = list(repeat_items(original, self.__num_samples))

        original_sentences = [x[self.original_field] for x in repeated_items]
        negated = self.__neg_polyjuice(original_sentences)

        for orig, n in zip(repeated_items, negated):
            if n is None:
                continue
            if self.perturbations_field not in orig:
                orig[self.perturbations_field] = {}
            orig[self.perturbations_field][self.critical_field] = n

        return repeated_items
