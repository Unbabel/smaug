import functools
import numpy as np
import re
import stanza
import transformers

from smaug import more_functools
from smaug import pipeline
from smaug import ops
from smaug import transform
from smaug import validation
from smaug.core import Data, DataLike

_SWP_NUM_PERTURBATION = "transf-swp-num"
_SWP_NE_PERTURBATION = "transf-swp-ne"


def perturb_swap_num(
    records: DataLike[pipeline.State],
    mT5_model: transformers.MT5ForConditionalGeneration,
    mT5_tokenizer: transformers.T5Tokenizer,
    rng: np.random.Generator,
    gpu: bool = False,
    default_validations: bool = True,
) -> Data[pipeline.State]:
    """Swaps a number in the received sentences.

    It searches for numbers in the original records using a regular expression and
    then uses Google's mT5 to replace the one of the found expressions with text.

    It can also ensure that the generated sentences are not equal to the original
    sentences and that they have the same number of numbers (by using the same
    regular expression).

    Args:
        records: Records to perturb.
        mT5_model: mT5 model to use.
        mT5_tokenizer: mT5 tokenizer to use.
        rng: Numpy random generator to use.
        gpu: Whether to use gpu.
        default_validations: Whether to run default validations to ensure some
        basic record quality.

    Returns:
        The perturbed records.
    """

    mask_func = functools.partial(
        ops.mask_detections,
        detect_func=ops.regex_detect_numbers,
        mask_func=ops.mT5_masking_function,
        rng=rng,
        max_masks=1,
    )

    fill_func = functools.partial(
        ops.mT5_generate, model=mT5_model, tokenizer=mT5_tokenizer, cuda=gpu
    )

    transform_func = functools.partial(
        transform.mask_and_fill,
        perturbation=_SWP_NUM_PERTURBATION,
        mask_func=mask_func,
        fill_func=fill_func,
    )

    if not default_validations:
        return transform_func(records)

    not_equal_func = functools.partial(
        validation.not_equal,
        perturbation=_SWP_NUM_PERTURBATION,
    )

    rm_masks_func = functools.partial(
        validation.no_regex_match,
        perturbation=_SWP_NUM_PERTURBATION,
        pattern=re.compile(r"<extra_id_\d{1,2}>"),
    )

    rm_symbols_insertion_func = functools.partial(
        validation.leq_char_insertions,
        perturbation=_SWP_NUM_PERTURBATION,
        chars="<>()[]{}_",
        max_insertions=0,
    )

    eq_num_count_func = functools.partial(
        validation.equal_numbers_count,
        perturbation=_SWP_NUM_PERTURBATION,
    )

    return more_functools.pipe(
        transform_func,
        not_equal_func,
        rm_masks_func,
        rm_symbols_insertion_func,
        eq_num_count_func,
    )(records)


def perturb_swap_named_entity(
    records: DataLike[pipeline.State],
    ner_pipeline: stanza.Pipeline,
    mT5_model: transformers.MT5ForConditionalGeneration,
    mT5_tokenizer: transformers.T5Tokenizer,
    rng: np.random.Generator,
    gpu: bool = False,
    default_validations: bool = True,
) -> Data[pipeline.State]:
    """Swaps a named entity in the received sentences.

    It searches for named entities in the original records using a ner model and
    then uses Google's mT5 to replace the one of the found expressions with text.

    It can also ensure that the generated sentences are not equal to the original
    sentences and that they have the same number of named entities (by using the
    same NER model).

    Args:
        records: Records to perturb.
        ner_pipeline: stanza NER pipeline to user.
        mT5_model: mT5 model to use.
        mT5_tokenizer: mT5 tokenizer to use.
        rng: Numpy random generator to use.
        gpu: Whether to use gpu.
        default_validations: Whether to run default validations to ensure some
        basic record quality.

    Returns:
        The perturbed records.
    """
    ner_func = functools.partial(
        ops.stanza_detect_named_entities,
        ner_pipeline=ner_pipeline,
    )

    mask_func = functools.partial(
        ops.mask_detections,
        detect_func=ner_func,
        mask_func=ops.mT5_masking_function,
        rng=rng,
        max_masks=1,
    )

    fill_func = functools.partial(
        ops.mT5_generate, model=mT5_model, tokenizer=mT5_tokenizer, cuda=gpu
    )

    transform_func = functools.partial(
        transform.mask_and_fill,
        perturbation=_SWP_NE_PERTURBATION,
        mask_func=mask_func,
        fill_func=fill_func,
    )

    if not default_validations:
        return transform_func(records)

    not_equal_func = functools.partial(
        validation.not_equal,
        perturbation=_SWP_NE_PERTURBATION,
    )

    rm_masks_func = functools.partial(
        validation.no_regex_match,
        perturbation=_SWP_NE_PERTURBATION,
        pattern=re.compile(r"<extra_id_\d{1,2}>"),
    )

    rm_symbols_insertion_func = functools.partial(
        validation.leq_char_insertions,
        perturbation=_SWP_NE_PERTURBATION,
        chars="<>()[]{}_",
        max_insertions=0,
    )

    eq_ne_count_func = functools.partial(
        validation.equal_named_entites_count,
        perturbation=_SWP_NE_PERTURBATION,
        ner_func=ner_func,
    )

    return more_functools.pipe(
        transform_func,
        not_equal_func,
        rm_masks_func,
        rm_symbols_insertion_func,
        eq_ne_count_func,
    )(records)
