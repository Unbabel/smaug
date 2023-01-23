import numpy as np
import re
import transformers

from smaug import functional
from smaug import ops
from smaug.core import Data, DataLike, Sentence, SentenceLike

from typing import Optional


def swap_poisson_span(
    sentences: DataLike[SentenceLike],
    mt5_model: transformers.MT5ForConditionalGeneration,
    mt5_tokenizer: transformers.T5Tokenizer,
    rng: np.random.Generator,
    gpu: bool = False,
) -> Data[Optional[Sentence]]:
    """Replaces a text span with size determined by the Poisson distribution.

    This perturbation masks a text span with size determined by the Poisson
    distribution and then uses Google's mT5 to fill the mask.

    It also runs default validations to ensure a minimum quality
    level.

    Args:
        sentences: Sentences to transform.
        mt5_model: mT5 model to use for generation.
        mt5_tokenizer: mT5 tokenizer for generation.
        rng: Numpy random generator to use.
        gpu: Whether to use gpu.

    Returns:
        Perturbed sentences. Returns None for sentences for which
        the validations failed.
    """
    transformed = swap_poisson_span_transform(
        sentences,
        mt5_model,
        mt5_tokenizer,
        rng,
        gpu,
    )
    return swap_poisson_span_validation(sentences, transformed)


def swap_poisson_span_transform(
    sentences: DataLike[SentenceLike],
    mt5_model: transformers.MT5ForConditionalGeneration,
    mt5_tokenizer: transformers.T5Tokenizer,
    rng: np.random.Generator,
    gpu: bool = False,
) -> Data[Sentence]:
    """Performs the transform phase for the swap_poisson_span perturbation.

    Args:
        sentences: Sentences to transform.
        mt5_model: mT5 model to use for generation.
        mt5_tokenizer: mT5 tokenizer for generation.
        rng: Numpy random generator to use.
        gpu: Whether to use gpu.

    Returns:
        Transformed sentences.
    """
    masked = ops.mask_poisson_spans(
        sentences,
        func=ops.mT5_masking_function,
        rng=rng,
    )

    return ops.mT5_generate(
        masked,
        model=mt5_model,
        tokenizer=mt5_tokenizer,
        cuda=gpu,
    )


def swap_poisson_span_validation(
    originals: DataLike[SentenceLike],
    transformed: DataLike[Optional[SentenceLike]],
) -> Data[Optional[Sentence]]:
    """Performs the validation phase for the swap_poisson_span perturbation.

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
