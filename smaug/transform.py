import numpy as np
import re

from typing import Callable, Optional

from smaug import ops
from smaug import pipeline
from smaug.core import Data, DataLike, Sentence, SentenceLike
from smaug.promote import promote_to_data, promote_to_sentence


def random_delete(
    records: DataLike[pipeline.State],
    perturbation: str,
    rng: np.random.Generator,
    p: float = 0.2,
) -> Data[pipeline.State]:
    """Deletes random words in the sentences.

    Args:
        records: Records to transform.
        perturbation: Name of the perturbation to consider.
        rng: Numpy generator to use.
        p: Probability of deleting a word.

    Returns:
        Transformed records.
    """

    def next_word_start(s: Sentence, start: int):
        # Try to find next space
        word_delim_idx = ops.find(s, " ", start=start)
        if word_delim_idx == -1:
            # If not space, then we are at the last word
            # and return the remaining sentence.
            word_delim_idx = len(s)
        return word_delim_idx + 1

    def transform(s: SentenceLike) -> Sentence:
        s = promote_to_sentence(s)

        curr_idx = 0
        while curr_idx < len(s):
            word_start_idx = next_word_start(s, curr_idx)
            if rng.random() < p:
                s = ops.delete(s, (curr_idx, word_start_idx))
            else:
                curr_idx = word_start_idx

        return s

    return _transform_with_func(records, perturbation, transform)


def punct_span_delete(
    records: DataLike[pipeline.State],
    perturbation: str,
    rng: np.random.Generator,
    punct: str = ".,!?",
    low: int = 4,
    high: int = 10,
) -> Data[pipeline.State]:
    """Removes a span between two punctuation symbols.

    Args:
        records: Records to transform.
        perturbation: Name of the perturbation to consider.
        rng: Numpy random generator to use.
        punct: Punctuation symbols to consider.
        low: Minimum number of words for a span to be eligible for deletion.
        high: Maximum number of words for a span to be eligible for deletion.
    """

    def transform(s: SentenceLike) -> Optional[Sentence]:
        s = promote_to_sentence(s)

        matches = punct_regex.finditer(s.value)
        spans_delims_idxs = [0] + [m.end() for m in matches] + [len(s)]
        # Transform indexes in iterable with (idx1,idx2), (idx2,idx3), ...
        pairwise = zip(spans_delims_idxs, spans_delims_idxs[1:])

        possible_spans_idxs = [
            (start, end)
            for start, end in pairwise
            if start > 0 and low < len(s.value[start:end].split()) < high
        ]
        if len(possible_spans_idxs) == 0:
            return None

        idx_to_drop = rng.choice(possible_spans_idxs)

        s = ops.rstrip(ops.delete(s, idx_to_drop))

        # To increase credibility of generated sentence,
        # replace last "," with "." .
        if not ops.endswith(s, (".", "?", "!")):
            s = ops.delete(s, (len(s) - 1, len(s)))
            s = ops.append(s, ".")

        return s

    punct_regex = re.compile(f"[{punct}]+")
    return _transform_with_func(records, perturbation, transform)


def negate(
    records: DataLike[pipeline.State],
    perturbation: str,
    polyjuice_func: Callable[[DataLike[SentenceLike]], Data[Optional[Sentence]]],
) -> Data[pipeline.State]:
    """Negates the original sentence.

    Not all sentences can be negated, and so only some records will be updated.

    Args:
        records: Records to transform.
        perturbation: Name of the perturbation to apply.
        polyjuice_func: Polyjuice function for negation.

    Returns:
        The transformed records.
    """
    records = promote_to_data(records)
    original_sentences = [x.original for x in records]
    negated = polyjuice_func(original_sentences)

    for orig, n in zip(records, negated):
        if n is None:
            continue
        orig.perturbations[perturbation] = n.value
        if n.trace is not None:
            orig.metadata[perturbation] = ops.modified_spans_from_trace(n.trace)

    return records


def mask_and_fill(
    records: DataLike[pipeline.State],
    perturbation: str,
    mask_func: Callable[[DataLike[SentenceLike]], Data[Sentence]],
    fill_func: Callable[[DataLike[SentenceLike]], Data[Sentence]],
) -> Data[pipeline.State]:
    """Generates critical errors by masking and filling sentences.

    Args:
        records: Records to transform.
        perturbation: Name of the perturbation to apply.
        mask_func: Mask function to use when masking sentences.
        fill_func: Masked language model to fill the masks in the sentences.

    Returns:
        The transformed records.
    """
    records = promote_to_data(records)
    original_sentences = Data(x.original for x in records)
    masked = mask_func(original_sentences)
    filled = fill_func(masked)

    for orig, t in zip(records, filled):
        orig.perturbations[perturbation] = t.value
        if t.trace is not None:
            orig.metadata[perturbation] = ops.modified_spans_from_trace(t.trace)

    return records


def _transform_with_func(
    records: DataLike[pipeline.State],
    perturbation: str,
    transf_func: Callable[[SentenceLike], Optional[Sentence]],
) -> Data[pipeline.State]:
    """Transforms records by applying a given transform function.

    The transform function may not be successfull, in which case it
    should return None.

    Args:
        records: Records to transform.
        perturbation: Name of the perturbation to consider.
        transf_func: Transform function.

    Returns:
        Transformed records.
    """
    records = promote_to_data(records)
    for orig in records:
        perturbed = transf_func(orig.original)
        if perturbed is not None:
            orig.perturbations[perturbation] = perturbed
            if perturbed.trace is not None:
                orig.metadata[perturbation] = ops.modified_spans_from_trace(
                    perturbed.trace
                )
    return records
