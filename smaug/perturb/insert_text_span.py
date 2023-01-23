import numpy as np
import re
import transformers

from smaug import functional
from smaug import ops
from smaug.core import Data, DataLike, Sentence, SentenceLike

from typing import Optional


def insert_text_span(
    sentences: DataLike[SentenceLike],
    mt5_model: transformers.MT5ForConditionalGeneration,
    mt5_tokenizer: transformers.T5Tokenizer,
    rng: np.random.Generator,
    p: float = 0.1,
    max_masks: int = 3,
    gpu: bool = False,
) -> Data[Optional[Sentence]]:
    """Inserts spans of text at random places in the sentence.

    This perturbation inserts masks at random places in the
    sentence and then uses mT5 to create new content.

    It also runs default validations to ensure a minimum quality
    level.

    Args:
        sentences: Sentences to transform.
        mt5_model: mT5 model to use for generation.
        mt5_tokenizer: mT5 tokenizer for generation.
        rng: Numpy random generator to use.
        p: Probability of inserting a mask between two words.
        max_masks: Maximum number of masks to insert.
        gpu: Whether to use gpu.

    Returns:
         Perturbed sentences. Returns None for sentences for which
         the validations failed.
    """
    transformed = insert_text_span_transform(
        sentences,
        mt5_model,
        mt5_tokenizer,
        rng,
        p,
        max_masks,
        gpu,
    )
    return insert_text_span_validation(sentences, transformed)


def insert_text_span_transform(
    sentences: DataLike[SentenceLike],
    mt5_model: transformers.MT5ForConditionalGeneration,
    mt5_tokenizer: transformers.T5Tokenizer,
    rng: np.random.Generator,
    p: float = 0.1,
    max_masks: int = 3,
    gpu: bool = False,
) -> Data[Sentence]:
    """Performs the transform phase of the insert_text_span perturbation.

    Args:
        sentences: Sentences to transform.
        mt5_model: mT5 model to use for generation.
        mt5_tokenizer: mT5 tokenizer for generation.
        rng: Numpy random generator to use.
        p: Probability of inserting a mask between two words.
        max_masks: Maximum number of masks to insert.
        gpu: Whether to use gpu.

    Returns:
        Transformed sentences.
    """
    masked = ops.mask_random_insert(
        sentences,
        func=ops.mT5_masking_function,
        rng=rng,
        p=p,
        max_masks=max_masks,
    )

    return ops.mT5_generate(
        masked,
        model=mt5_model,
        tokenizer=mt5_tokenizer,
        cuda=gpu,
    )


def insert_text_span_validation(
    originals: DataLike[SentenceLike],
    transformed: DataLike[Optional[SentenceLike]],
) -> Data[Optional[Sentence]]:
    """Performs basic validation for the insert_text_span perturbation.

    It validates that the generated sentences are different from
    the original, and ensures a basic quality level by removing
    sentences that match the mT5 masking pattern (<extra_id_\\d{1,2}>)
    and sentences with character insertions for <>()[]{}_, as they are
    likely model hallucinations.

    Args:
        originals: Original sentences.
        transformed: Transformed sentences.

    Returns:
        Validated sentences. Returns None for sentences for which
        the validations failed.
    """

    def val_func(o: Sentence, p: Sentence) -> bool:
        return (
            o != p
            and re.search(r"<extra_id_\d{1,2}>", p.value) is None
            and ops.character_insertions(o, p, "<>()[]{}_") == 0
        )

    return functional.lift_boolean_validation(val_func)(originals, transformed)
