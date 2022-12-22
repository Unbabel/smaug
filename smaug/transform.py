import numpy as np
import re

from typing import Callable, Optional

from smaug import core
from smaug import pipeline
from smaug import sentence


def random_delete(
    records: core.DataLike[pipeline.State],
    perturbation: str,
    rng: np.random.Generator,
    p: float = 0.2,
) -> core.Data[pipeline.State]:
    """Deletes random words in the sentences.

    Args:
        records: Records to transform.
        perturbation: Name of the perturbation to consider.
        rng: Numpy generator to use.
        p: Probability of deleting a word.

    Returns:
        Transformed records.
    """

    def next_word_start(s: sentence.Sentence, start: int):
        # Try to find next space
        word_delim_idx = s.find(" ", start=start)
        if word_delim_idx == -1:
            # If not space, then we are at the last word
            # and return the remainig sentence.
            word_delim_idx = len(s)
        return word_delim_idx + 1

    def transform(s: sentence.SentenceLike) -> sentence.Sentence:
        s = sentence.promote_to_sentence(s)

        curr_idx = 0
        while curr_idx < len(s):
            word_start_idx = next_word_start(s, curr_idx)
            if rng.random() < p:
                s = s.delete((curr_idx, word_start_idx))
            else:
                curr_idx = word_start_idx

        return s

    return _transform_with_func(records, perturbation, transform)


def punct_span_delete(
    records: core.DataLike[pipeline.State],
    perturbation: str,
    rng: np.random.Generator,
    punct: str = ".,!?",
    low: int = 4,
    high: int = 10,
) -> core.Data[pipeline.State]:
    """Removes a span between two punctuation symbols.

    Args:
        records: Records to transform.
        perturbation: Name of the perturbation to consider.
        rng: Numpy random generator to use.
        punct: Punctuation symbols to consider.
        low: Minimum number of words for a span to be eligible for deletion.
        high: Maximum number of words for a span to be eligible for deletion.
    """

    def transform(s: sentence.SentenceLike) -> Optional[sentence.Sentence]:
        s = sentence.promote_to_sentence(s)

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

        s = s.delete(idx_to_drop).rstrip()

        # Too increase credibility of generated sentence,
        # replace last "," with "." .
        if not s.endswith((".", "?", "!")):
            s = s.delete((len(s) - 1, len(s))).append(".")

        return s

    punct_regex = re.compile(f"[{punct}]+")
    return _transform_with_func(records, perturbation, transform)


def negate(
    records: core.DataLike[pipeline.State],
    perturbation: str,
    polyjuice_func: Callable[
        [core.DataLike[sentence.SentenceLike]], core.Data[Optional[sentence.Sentence]]
    ],
) -> core.Data[pipeline.State]:
    """Negates the original sentence.

    Not all sentences can be negated, and so only some records will be updated.

    Args:
        records: Records to transform.
        perturbation: Name of the perturbation to apply.
        polyjuice_func: Polyjuice function for negation.

    Returns:
        The transformed records.
    """
    records = core.promote_to_data(records)
    original_sentences = [x.original for x in records]
    negated = polyjuice_func(original_sentences)

    for orig, n in zip(records, negated):
        if n is None:
            continue
        orig.perturbations[perturbation] = n.value
        if n.trace is not None:
            orig.metadata[perturbation] = n.trace.modified_indices().compress()

    return records


def mask_and_fill(
    records: core.DataLike[pipeline.State],
    perturbation: str,
    mask_func: Callable[
        [core.DataLike[sentence.SentenceLike]], core.Data[sentence.Sentence]
    ],
    fill_func: Callable[
        [core.DataLike[sentence.SentenceLike]], core.Data[sentence.Sentence]
    ],
) -> core.Data[pipeline.State]:
    """Generates critical errors by masking and filling sentences.

    Args:
        records: Records to transform.
        perturbation: Name of the perturbation to apply.
        mask_func: Mask function to use when masking sentences.
        fill_func: Masked language model to fill the masks in the sentences.

    Returns:
        The transformed records.
    """
    records = core.promote_to_data(records)
    original_sentences = core.Data(x.original for x in records)
    masked = mask_func(original_sentences)
    filled = fill_func(masked)

    for orig, t in zip(records, filled):
        orig.perturbations[perturbation] = t.value
        if t.trace is not None:
            orig.metadata[perturbation] = t.trace.modified_indices().compress()

    return records


def _transform_with_func(
    records: core.DataLike[pipeline.State],
    perturbation: str,
    transf_func: Callable[[sentence.SentenceLike], Optional[sentence.Sentence]],
) -> core.Data[pipeline.State]:
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
    records = core.promote_to_data(records)
    for orig in records:
        perturbed = transf_func(orig.original)
        if perturbed is not None:
            orig.perturbations[perturbation] = perturbed
            if perturbed.trace is not None:
                orig.metadata[
                    perturbation
                ] = perturbed.trace.modified_indices().compress()
    return records
