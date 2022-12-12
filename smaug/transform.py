import io
import itertools
import numpy as np
import re

from typing import Callable, Iterable, List, Optional

from smaug import core
from smaug import pipeline
from smaug.ops import lang_model


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

    def transform(sentence: str) -> str:
        splits = sentence.split()
        return " ".join(filter(lambda _: rng.random() < keep_p, splits))

    keep_p = 1 - p
    return _transform_with_func(records, perturbation, transform)


def span_delete(
    records: core.DataLike[pipeline.State],
    perturbation: str,
    rng: np.random.Generator,
    min_size: float = 0.25,
) -> core.Data[pipeline.State]:
    """Deletes a random span of words in the sentences.

    Args:
        records: Records to transform.
        perturbation: Name of the perturbation to consider.
        rng: Numpy generator to use.
        min_size: Minimum number of words to delete.

    Returns:
        Transformed records.
    """

    def transform(sentence: str) -> str:
        splits = sentence.split()
        num_splits = len(splits)

        lower_idx, higher_idx = 0, 0
        span_size = higher_idx - lower_idx
        while span_size / num_splits <= min_size:
            lower_idx, higher_idx = rng.choice(
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

    def transform(sentence: str) -> Optional[str]:
        spans = punct_regex.split(sentence)
        # Indexes of spans that can be dropped.
        # The first index is not considered as models are
        # more likelly to fail on the end of the sentence.
        possible_drop_idxs = [
            i for i, s in enumerate(spans) if i > 0 and low < len(s.split()) < high
        ]
        # Only delete when there are several subsentences,
        # to avoid deleting the entire content, making the
        # example trivial to identify.
        if len(possible_drop_idxs) < 2:
            return None

        idx_to_drop = rng.choice(possible_drop_idxs)
        buffer = io.StringIO()
        sentence_idx = 0

        for i, span in enumerate(spans):
            if i != idx_to_drop:
                buffer.write(span)
            sentence_idx += len(span)

            if i < len(spans) - 1:
                punct_after_span = punct_regex.match(sentence, pos=sentence_idx)
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

    punct_regex = re.compile(f"[{punct}]+")
    return _transform_with_func(records, perturbation, transform)


def shuffle_named_entities(
    records: core.DataLike[pipeline.State],
    perturbation: str,
    entities: List[str],
    ner_func: Callable[[core.DataLike[str]], core.Data],
    rng: np.random.Generator,
) -> core.Data[pipeline.State]:
    """Shuffles the named entities in a sentence.

    This transform uses a NER model to identify entities according to several
    tags and then shuffles entities with the same tag.

    Args:
        records: Records to transform.
        perturbatinon: Name of the perturbation to consider.
        entities: Entity tags to consider. They should be a subset of the tags
        supported by the ner_func.
        ner_func: Function to perform NER.
        rng: numpy generator to use.

    Returns:
        The transformed records.
    """

    def shuffle_single_entity_type(entity: str, sentence: str):
        entities = ner_func(sentence).item().entities
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
        rng.shuffle(swapped_indexes)
        while np.all(np.equal(ordered_indexes, swapped_indexes)):
            rng.shuffle(swapped_indexes)

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

    def shuffle_all_entity_types(
        entities: Iterable[str], sentence: str
    ) -> Optional[str]:
        current = sentence

        for entity in entities:
            # Continue if more than two entities were detected in total.
            current, should_continue = shuffle_single_entity_type(
                entity,
                current,
            )
            if not should_continue:
                break

        return current

    def transform(sentence: str) -> Optional[str]:
        return shuffle_all_entity_types(entities, sentence)

    return _transform_with_func(records, perturbation, transform)


def negate(
    records: core.DataLike[pipeline.State],
    perturbation: str,
    polyjuice_func: Callable[[core.DataLike[str]], core.Data[Optional[str]]],
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
        orig.perturbations[perturbation] = n

    return records


def mask_and_fill(
    records: core.DataLike[pipeline.State],
    perturbation: str,
    mask_func: Callable[[core.DataLike[str]], core.Data[str]],
    fill_func: Callable[[core.DataLike[str]], lang_model.MaskedLanguageModelOutput],
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
    original_sentences = [x.original for x in records]
    masked = mask_func(original_sentences)
    filled = fill_func(masked)

    for orig, t in zip(records, filled.text):
        orig.perturbations[perturbation] = t

    for orig, s in zip(records, filled.spans):
        orig.metadata[perturbation] = s

    return records


def _transform_with_func(
    records: core.DataLike[pipeline.State],
    perturbation: str,
    transf_func: Callable[[str], Optional[str]],
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
    return records
