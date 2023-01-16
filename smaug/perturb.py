import functools
import numpy as np
import re
import stanza
import transformers

from smaug import more_functools
from smaug import pipeline
from smaug import ops
from smaug import validation
from smaug.core import Data, DataLike

_SWP_NUM_PERTURBATION = "transf-swp-num"
_SWP_NE_PERTURBATION = "transf-swp-ne"
_SWP_POISSON_SPAN_PERTURBATION = "transf-swp-poisson-span"
_NEG_PERTURBATION = "transf-neg"
_INS_TEXT_SPAN_PERTURBATION = "transf-ins-text"
_DEL_PUNCTUATION_SPAN_PERTURBATION = "transf-del-punct-span"


def swap_number(
    records: DataLike[pipeline.State],
    mt5_model: transformers.MT5ForConditionalGeneration,
    mt5_tokenizer: transformers.T5Tokenizer,
    rng: np.random.Generator,
    gpu: bool = False,
) -> Data[pipeline.State]:
    """Swaps a number in the received sentences.

    It searches for numbers in the original records using a regular expression and
    then uses Google's mt5 to replace the one of the found expressions with text.

    It also ensures that the generated sentences are not equal to the original
    sentences and that they have the same number of numbers (by using the same
    regular expression).

    Args:
        records: Records to perturb.
        mt5_model: mt5 model to use.
        mt5_tokenizer: mt5 tokenizer to use.
        rng: Numpy random generator to use.
        gpu: Whether to use gpu.

    Returns:
        The perturbed records.
    """

    transformed = swap_number_transform(
        records,
        mt5_model,
        mt5_tokenizer,
        rng,
        gpu,
    )
    return swap_number_validation(transformed)


def swap_number_validation(records: DataLike[pipeline.State]) -> Data[pipeline.State]:
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
        not_equal_func,
        rm_masks_func,
        rm_symbols_insertion_func,
        eq_num_count_func,
    )(records)


def swap_named_entity(
    records: DataLike[pipeline.State],
    ner_pipeline: stanza.Pipeline,
    mt5_model: transformers.MT5ForConditionalGeneration,
    mt5_tokenizer: transformers.T5Tokenizer,
    rng: np.random.Generator,
    gpu: bool = False,
) -> Data[pipeline.State]:
    """Swaps a named entity in the received sentences.

    It searches for named entities in the original records using a ner model and
    then uses Google's mt5 to replace the one of the found expressions with text.

    It also ensures that the generated sentences are not equal to the original
    sentences and that they have the same number of named entities (by using the
    same NER model).

    Args:
        records: Records to perturb.
        ner_pipeline: stanza NER pipeline to user.
        mt5_model: mt5 model to use.
        mt5_tokenizer: mt5 tokenizer to use.
        rng: Numpy random generator to use.
        gpu: Whether to use gpu.

    Returns:
        The perturbed records.
    """

    transformed = swap_named_entity_transform(
        records, ner_pipeline, mt5_model, mt5_tokenizer, rng, gpu
    )
    return swap_named_entity_validation(transformed, ner_pipeline)


def swap_named_entity_validation(
    records: DataLike[pipeline.State],
    ner_pipeline: stanza.Pipeline,
) -> Data[pipeline.State]:
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

    ner_func = functools.partial(
        ops.stanza_detect_named_entities,
        ner_pipeline=ner_pipeline,
    )

    eq_ne_count_func = functools.partial(
        validation.equal_named_entites_count,
        perturbation=_SWP_NE_PERTURBATION,
        ner_func=ner_func,
    )

    return more_functools.pipe(
        not_equal_func,
        rm_masks_func,
        rm_symbols_insertion_func,
        eq_ne_count_func,
    )(records)


def swap_poisson_span(
    records: DataLike[pipeline.State],
    mt5_model: transformers.MT5ForConditionalGeneration,
    mt5_tokenizer: transformers.T5Tokenizer,
    rng: np.random.Generator,
    gpu: bool = False,
) -> Data[pipeline.State]:
    transformed = swap_poisson_span_transform(
        records,
        mt5_model,
        mt5_tokenizer,
        rng,
        gpu,
    )
    return swap_poisson_span_validation(transformed)


def swap_poisson_span_validation(
    records: DataLike[pipeline.State],
) -> Data[pipeline.State]:
    not_equal_func = functools.partial(
        validation.not_equal,
        perturbation=_SWP_POISSON_SPAN_PERTURBATION,
    )

    rm_masks_func = functools.partial(
        validation.no_regex_match,
        perturbation=_SWP_POISSON_SPAN_PERTURBATION,
        pattern=re.compile(r"<extra_id_\d{1,2}>"),
    )

    rm_symbols_insertion_func = functools.partial(
        validation.leq_char_insertions,
        perturbation=_SWP_POISSON_SPAN_PERTURBATION,
        chars="<>()[]{}_",
        max_insertions=0,
    )

    return more_functools.pipe(
        not_equal_func,
        rm_masks_func,
        rm_symbols_insertion_func,
    )(records)


def negate(
    records: DataLike[pipeline.State],
    pos_pipeline: stanza.Pipeline,
    polyjuice_model: transformers.AutoModelForCausalLM,
    polyjuice_tokenizer: transformers.PreTrainedTokenizerBase,
    roberta_model: transformers.RobertaForSequenceClassification,
    roberta_tokenizer: transformers.PreTrainedTokenizerBase,
    rng: np.random.Generator,
    gpu: bool = False,
) -> Data[pipeline.State]:
    transformed = negate_transform(
        records,
        pos_pipeline,
        polyjuice_model,
        polyjuice_tokenizer,
        rng,
        gpu,
    )
    return negate_validation(transformed, roberta_model, roberta_tokenizer, gpu)


def negate_validation(
    records: DataLike[pipeline.State],
    roberta_model: transformers.RobertaForSequenceClassification,
    roberta_tokenizer: transformers.PreTrainedTokenizerBase,
    gpu: bool = False,
):
    not_equal_func = functools.partial(
        validation.not_equal,
        perturbation=_NEG_PERTURBATION,
    )

    rm_equal = functools.partial(
        validation.no_regex_match,
        perturbation=_NEG_PERTURBATION,
        pattern=re.compile("EMPTY"),
    )

    roberta_predict_func = functools.partial(
        ops.roberta_mnli_predict,
        model=roberta_model,
        tokenizer=roberta_tokenizer,
        cuda=gpu,
    )
    roberta_contradiction_id = ops.roberta_mnli_contradiction_id(roberta_model)

    is_contradiction_func = functools.partial(
        validation.is_contradiction,
        perturbation=transform,
        predict_func=roberta_predict_func,
        contradiction_id=roberta_contradiction_id,
    )

    return more_functools.pipe(
        not_equal_func,
        rm_equal,
        is_contradiction_func,
    )(records)


def insert_text_span(
    records: DataLike[pipeline.State],
    mt5_model: transformers.MT5ForConditionalGeneration,
    mt5_tokenizer: transformers.T5Tokenizer,
    rng: np.random.Generator,
    p: float = 0.1,
    max_masks: int = 3,
    gpu: bool = False,
) -> Data[pipeline.State]:
    transformed = insert_text_span_transform(
        records,
        mt5_model,
        mt5_tokenizer,
        rng,
        p,
        max_masks,
        gpu,
    )
    return insert_text_span_validation(transformed)


def insert_text_span_validation(
    records: DataLike[pipeline.State],
) -> Data[pipeline.State]:
    not_equal_func = functools.partial(
        validation.not_equal,
        perturbation=_INS_TEXT_SPAN_PERTURBATION,
    )

    rm_masks_func = functools.partial(
        validation.no_regex_match,
        perturbation=_INS_TEXT_SPAN_PERTURBATION,
        pattern=re.compile(r"<extra_id_\d{1,2}>"),
    )

    rm_symbols_insertion_func = functools.partial(
        validation.leq_char_insertions,
        perturbation=_INS_TEXT_SPAN_PERTURBATION,
        chars="<>()[]{}_",
        max_insertions=0,
    )

    return more_functools.pipe(
        not_equal_func,
        rm_masks_func,
        rm_symbols_insertion_func,
    )(records)
