import functools
import numpy as np
import re
import stanza
import transformers

from smaug import ops
from smaug import functional
from smaug.core import Data, DataLike, Sentence, SentenceLike

from typing import Optional


def swap_named_entity(
    sentences: DataLike[SentenceLike],
    ner_pipeline: stanza.Pipeline,
    mt5_model: transformers.MT5ForConditionalGeneration,
    mt5_tokenizer: transformers.T5Tokenizer,
    rng: np.random.Generator,
    gpu: bool = False,
) -> Data[Optional[Sentence]]:
    """Swaps a named entity in the received sentences.

    It searches for named entities in the original records using a ner model and
    then uses Google's mt5 to replace the one of the found expressions with text.

    It also ensures that the generated sentences are not equal to the original
    sentences and that they have the same number of named entities (by using the
    same NER model).

    Args:
        sentences: Records to perturb.
        ner_pipeline: stanza NER pipeline to user.
        mt5_model: mt5 model to use.
        mt5_tokenizer: mt5 tokenizer to use.
        rng: Numpy random generator to use.
        gpu: Whether to use gpu.

    Returns:
        The perturbed records.
    """

    transformed = swap_named_entity_transform(
        sentences, ner_pipeline, mt5_model, mt5_tokenizer, rng, gpu
    )
    return swap_named_entity_validation(sentences, transformed, ner_pipeline)


def swap_named_entity_transform(
    sentences: DataLike[SentenceLike],
    ner_pipeline: stanza.Pipeline,
    mt5_model: transformers.MT5ForConditionalGeneration,
    mt5_tokenizer: transformers.T5Tokenizer,
    rng: np.random.Generator,
    gpu: bool = False,
) -> Data[Sentence]:
    ner_func = functools.partial(
        ops.stanza_detect_named_entities,
        ner_pipeline=ner_pipeline,
    )

    masked = ops.mask_detections(
        sentences,
        detect_func=ner_func,
        mask_func=ops.mT5_masking_function,
        rng=rng,
        max_masks=1,
    )

    return ops.mT5_generate(
        masked,
        model=mt5_model,
        tokenizer=mt5_tokenizer,
        cuda=gpu,
    )


def swap_named_entity_validation(
    originals: DataLike[SentenceLike],
    perturbations: DataLike[Optional[SentenceLike]],
    ner_pipeline: stanza.Pipeline,
) -> Data[Optional[Sentence]]:
    def val_func(o: Sentence, p: Sentence) -> bool:
        return (
            o != p
            and re.search(r"<extra_id_\d{1,2}>", p.value) is None
            and ops.character_insertions(o, p, "<>()[]{}_") == 0
            and ops.equal_named_entities_count(o, p, ner_pipeline)
        )

    return functional.lift_boolean_validation(val_func)(originals, perturbations)
