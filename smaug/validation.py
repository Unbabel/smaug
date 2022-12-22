import collections
import nltk.metrics
import re
import torch

from typing import Callable

from smaug import core
from smaug import sentence
from smaug import pipeline


def not_equal(
    records: core.DataLike[pipeline.State], perturbation: str
) -> core.Data[pipeline.State]:
    """Filters critical records that are equal to the original.

    Args:
        records: Records to validate.
        perturbation: Name of the perturbation to filter.

    Returns:
        Validated records.
    """

    def val_func(
        original: sentence.SentenceLike, perturbed: sentence.SentenceLike
    ) -> bool:
        o = sentence.promote_to_sentence(original)
        p = sentence.promote_to_sentence(perturbed)
        return o != p

    return _validate_with_func(records, perturbation, val_func)


def equal_named_entites_count(
    records: core.DataLike[pipeline.State],
    perturbation: str,
    ner_func: Callable[[core.DataLike[sentence.SentenceLike]], core.Data],
) -> core.Data[pipeline.State]:
    """Filters records that do not have the same named entity count.

    Args:
        records: Records to validate.
        perturbation: Name of the perturbation to consider.
        ner_func: Function to perform NER.

    Returns:
        Validated records.
    """

    def val_func(
        original: sentence.SentenceLike, perturbed: sentence.SentenceLike
    ) -> bool:
        orig_entity_count = len(ner_func(original).item().entities)
        pert_entity_count = len(ner_func(perturbed).item().entities)
        return orig_entity_count == pert_entity_count

    return _validate_with_func(records, perturbation, val_func)


_NUM_REGEX = re.compile(r"[-+]?\.?(\d+[.,])*\d+")


def equal_numbers_count(
    records: core.DataLike[pipeline.State], perturbation: str
) -> core.Data[pipeline.State]:
    """Filters records that do not have the same numbers count.

    Args:
        records: Records to validate.
        perturbation: Name of the perturbation to consider.

    Returns:
        Validated records.
    """

    def cmp_func(
        original: sentence.SentenceLike, perturbed: sentence.SentenceLike
    ) -> bool:
        o = sentence.promote_to_sentence(original)
        p = sentence.promote_to_sentence(perturbed)
        orig_count = len(_NUM_REGEX.findall(o.value))
        crit_count = len(_NUM_REGEX.findall(p.value))
        return orig_count == crit_count

    return _validate_with_func(records, perturbation, cmp_func)


def is_contradiction(
    records: core.DataLike[pipeline.State],
    perturbation: str,
    predict_func: Callable[[core.DataLike[sentence.SentenceLike]], torch.FloatTensor],
    contradiction_id: int,
) -> core.Data[pipeline.State]:
    """Filters perturbed records that do not contradict the original sentence.

    Args:
        records: Records to validate.
        perturbation: Name of the perturbation to consider.
        predict_func: Function to predict whether a sentence contradicts the other.
        contradiction_id: Id for the contradiction label.

    Returns:
        Validated records.
    """

    def val_func(
        original: sentence.SentenceLike, perturbed: sentence.SentenceLike
    ) -> bool:
        nli_input = f"{original} </s></s> {perturbed}"
        logits = predict_func(nli_input)
        predicted_id = logits.argmax().item()
        return predicted_id == contradiction_id

    return _validate_with_func(records, perturbation, val_func)


def min_relative_length(
    records: core.DataLike[pipeline.State], perturbation: str, threshold: float
) -> core.Data[pipeline.State]:
    """Filters critical records that are too short when compared to the original.

    Args:
        records: Records to validate.
        perturbation: Name of the perturbation to consider.
        threshold: Minimum ratio len(critical) / len(original) that should be accepted.

    Returns:
        Validated records.
    """
    return _validate_with_func(
        records, perturbation, lambda orig, crit: len(orig) / len(crit) >= threshold
    )


def no_regex_match(
    records: core.DataLike[pipeline.State], perturbation: str, pattern: re.Pattern
) -> core.Data[pipeline.State]:
    """Excludes perturbed sentences that match a given regular expression.

    Args:
        records: Records to validate.
        perturbation: Name of the perturbation to consider.
        pattern: Pattern to search.

    Returns:
        Validated records.
    """

    def val_func(_: sentence.SentenceLike, perturbed: sentence.SentenceLike) -> bool:
        p = sentence.promote_to_sentence(perturbed)
        return pattern.search(p.value) is None

    return _validate_with_func(records, perturbation, val_func)


def geq_edit_distance(
    records: core.DataLike[pipeline.State], perturbation: str, min_dist: int, level: str
) -> core.Data[pipeline.State]:
    """Filters perturbations with a small minimum edit distance to the original.

    Args:
        records: Records to validate.
        perturbation: Name of the perturbation to consider.
        min_dist: Minimum edit distance that should be accepted.
        level: Level at which to measure the minimum edit distance. Must be "word" or "char".

    Raises:
        ValueError: If the level is not "word" or "char".

    Returns:
        Validated records.
    """

    def char_val_func(
        original: sentence.SentenceLike, perturbed: sentence.SentenceLike
    ) -> bool:
        o = sentence.promote_to_sentence(original)
        p = sentence.promote_to_sentence(perturbed)
        return nltk.metrics.edit_distance(o.value, p.value) >= min_dist

    def word_val_func(
        original: sentence.SentenceLike, perturbed: sentence.SentenceLike
    ) -> bool:
        o = sentence.promote_to_sentence(original)
        p = sentence.promote_to_sentence(perturbed)
        return nltk.metrics.edit_distance(o.value.split(), p.value.split()) >= min_dist

    levels = ("char", "word")
    if level not in levels:
        raise ValueError(f"Unknown level {level}: must be one of {levels}.")
    cmp_func = char_val_func if level == "char" else word_val_func
    return _validate_with_func(records, perturbation, cmp_func)


def leq_char_insertions(
    records: core.DataLike[pipeline.State],
    perturbation: str,
    chars: str,
    max_insertions: int,
) -> core.Data[pipeline.State]:
    """Filters perturbations with many insertions of specific characters when compared to the original.

    This validation takes a set of characters and adds up how many insertions of these charactes
    the perturbed sentence has. If this number is over a threshold, the perturbation is rejected.

    Args:
        records: Records to validate.
        perturbation: Name of the perturbation to consider.
        chars: String with characters to consider (each individual character will be considered).
        max_insertions: Maximum number of insertions.

    Returns:
        Validated records.
    """

    def cmp_func(
        original: sentence.SentenceLike, perturbed: sentence.SentenceLike
    ) -> bool:
        original_chars = (c for c in original if c in chars)
        perturbed_chars = (c for c in perturbed if c in chars)
        original_counts = collections.Counter(original_chars)
        perturbed_counts = collections.Counter(perturbed_chars)
        insertions = perturbed_counts - original_counts
        return sum(insertions.values()) <= max_insertions

    return _validate_with_func(records, perturbation, cmp_func)


def _validate_with_func(
    records: core.DataLike[pipeline.State],
    perturbation: str,
    val_func: Callable[[sentence.SentenceLike, sentence.SentenceLike], bool],
) -> core.Data[pipeline.State]:
    """Filters critical records by comparing with the original sentence.

    Args:
        records: Records to validate.
        perturbation: Name of the perturbation to consider.
        val_func: Function to validate whether a record should be accepted.

    Returns:
        Validated records.
    """
    records = core.promote_to_data(records)
    for r in records:
        if perturbation not in r.perturbations:
            continue
        if not val_func(r.original, r.perturbations[perturbation]):
            _del_perturbation(perturbation, r)
    return records


def _del_perturbation(field: str, state: pipeline.State):
    if field in state.perturbations:
        del state.perturbations[field]
    if field in state.metadata:
        del state.metadata[field]
