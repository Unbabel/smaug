import collections
import nltk.metrics
import re
import torch

from typing import Callable, List

from smaug import core
from smaug import pipeline


def cmp_based(
    records: List[pipeline.State],
    perturbation: str,
    cmp_func: Callable[[str, str], bool],
) -> List[pipeline.State]:
    """Filters critical records by comparing with the original sentence.

    Args:
        field (str): perturbation name to consider.
        cmp_func (Callable[[str, str], bool]): function to validate whether a record should be accepted.
        records (List[pipeline.State]): records to process.

    Returns:
        List[pipeline.State]: validated records.
    """
    for r in records:
        if perturbation not in r.perturbations:
            continue
        if not cmp_func(r.original, r.perturbations[perturbation]):
            del_perturbation(perturbation, r)
    return records


def del_perturbation(field: str, state: pipeline.State):
    if field in state.perturbations:
        del state.perturbations[field]
    if field in state.metadata:
        del state.metadata[field]


def equal_named_entites_count(
    records: List[pipeline.State],
    perturbation: str,
    ner_func: Callable[[core.DataLike[str]], core.Data],
) -> List[pipeline.State]:
    """Filters records that do not have the same named entity count.

    Args:
        perturbation (str): name of the perturbation to consider.
        ner_func (Callable[[core.DataLike[str]], core.Data]): function to perform NER.
        records (List[pipeline.State]): records to validate.

    Returns:
        List[pipeline.State]: validated records.
    """

    def cmp_func(original: str, critical: str) -> bool:
        orig_entity_count = len(ner_func(original).item().entities)
        crit_entity_count = len(ner_func(critical).item().entities)
        return orig_entity_count == crit_entity_count

    return cmp_based(records, perturbation, cmp_func)


def is_contradiction(
    records: List[pipeline.State],
    perturbation: str,
    predict_func: Callable[[core.DataLike[str]], torch.FloatTensor],
    contradiction_id: int,
) -> List[pipeline.State]:
    """Filters perturbed records that do not contradict the original sentence.

    Args:
        records (List[pipeline.State]): Records to validate.
        perturbation (str): Name of the perturbation to consider.
        predict_func (Callable[[core.DataLike[str]], torch.FloatTensor]): Function
        to predict whether a sentence contradicts the other.
        contradiction_id (int): Id for the contradiction label.

    Returns:
        List[pipeline.State]: Validated records.
    """
    for r in records:
        if perturbation not in r.perturbations:
            continue
        nli_input = f"{r.original} </s></s> {r.perturbations[perturbation]}"
        logits = predict_func(nli_input)
        predicted_id = logits.argmax().item()
        if predicted_id != contradiction_id:
            del_perturbation(perturbation, r)
    return records


_NUM_REGEX = re.compile(r"[-+]?\.?(\d+[.,])*\d+")


def equal_numbers_count(
    records: List[pipeline.State],
    perturbation: str,
) -> List[pipeline.State]:
    """Filters records that do not have the same numbers count.

    Args:
        records (List[pipeline.State]): Records to validate.
        perturbation (str): Name of the perturbation to consider.

    Returns:
        List[pipeline.State]: Validated records.
    """

    def cmp_func(original: str, critical: str) -> bool:
        orig_count = len(_NUM_REGEX.findall(original))
        crit_count = len(_NUM_REGEX.findall(critical))
        return orig_count == crit_count

    return cmp_based(records, perturbation, cmp_func)


def not_equal(records: List[pipeline.State], perturbation: str) -> List[pipeline.State]:
    """Filters critical records that are equal to the original.

    Args:
        records (List[pipeline.State]): Records to validate.
        perturbation (str): Name of the perturbation to filter.

    Returns:
        List[pipeline.State]: Validated records.
    """
    return cmp_based(records, perturbation, lambda orig, crit: orig != crit)


def min_relative_length(
    records: List[pipeline.State], perturbation: str, threshold: float
) -> List[pipeline.State]:
    """Filters critical records that are too short when compared to the original.

    Args:
        records (List[pipeline.State]): Records to validate.
        perturbation (str): Name of the perturbation to consider.
        threshold (float): Minimum ratio len(critical) / len(original) that should be accepted.

    Returns:
        List[pipeline.State]: Validated records.
    """
    return cmp_based(
        records, perturbation, lambda orig, crit: len(orig) / len(crit) >= threshold
    )


def no_regex_match(
    records: List[pipeline.State], perturbation: str, pattern: re.Pattern
) -> List[pipeline.State]:
    """Excludes perturbed sentences that match a given regular expression.

    Args:
        records (List[pipeline.State]): Records to validate.
        perturbation (str): Name of the perturbation to consider.
        pattern (re.Pattern): Pattern to search.

    Returns:
        List[pipeline.State]: Validated records.
    """
    for r in records:
        if perturbation not in r.perturbations:
            continue
        if pattern.search(r.perturbations[perturbation]) is not None:
            del_perturbation(perturbation, r)
    return records


def geq_edit_distance(
    records: List[pipeline.State], perturbation: str, min_dist: str, level: str
) -> List[pipeline.State]:
    """Filters perturbations with a small minimum edit distance to the original.

    Args:
        records (List[pipeline.State]): Records to validate.
        perturbation (str): Name of the perturbation to consider.
        min_dist (str): Minimum edit distance that should be accepted.
        level (str): Level at which to measure the minimum edit distance. Must be "word" or "char".

    Raises:
        ValueError: If the level is not "word" or "char".

    Returns:
        List[pipeline.State]: Validated records.
    """

    def cmp_func(original: str, perturbed: str) -> bool:
        if level == "word":
            original = original.split()
            perturbed = perturbed.split()
        return nltk.metrics.edit_distance(original, perturbed) >= min_dist

    levels = ("char", "word")
    if level not in levels:
        raise ValueError(f"Unknown level {level}: must be one of {levels}.")
    return cmp_based(records, perturbation, cmp_func)


def leq_char_insertions(
    records: List[pipeline.State], perturbation: str, chars: str, max_insertions: int
) -> List[pipeline.State]:
    """Filters perturbations with many insertions of specific characters when compared to the original.

    This validation takes a set of characters and adds up how many insertions of these charactes
    the perturbed sentence has. If this number is over a threshold, the perturbation is rejected.

    Args:
        records (List[pipeline.State]): Records to validate.
        perturbation (str): Name of the perturbation to consider.
        chars (str): String with characters to consider (each individual character will be considered).
        max_insertions (int): Maximum number of insertions.

    Returns:
        List[pipeline.State]: Validated records.
    """

    def cmp_func(original: str, perturbed: str) -> bool:
        original_chars = (c for c in original if c in chars)
        perturbed_chars = (c for c in perturbed if c in chars)
        original_counts = collections.Counter(original_chars)
        perturbed_counts = collections.Counter(perturbed_chars)
        insertions = perturbed_counts - original_counts
        return sum(insertions.values()) <= max_insertions

    return cmp_based(records, perturbation, cmp_func)
