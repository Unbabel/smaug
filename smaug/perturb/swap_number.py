import numpy as np
import re
import transformers

from smaug import functional
from smaug import ops
from smaug.core import Data, DataLike, Sentence, SentenceLike

from typing import Optional


def swap_number(
    sentences: DataLike[SentenceLike],
    mt5_model: transformers.MT5ForConditionalGeneration,
    mt5_tokenizer: transformers.T5Tokenizer,
    rng: np.random.Generator,
    gpu: bool = False,
) -> Data[Optional[Sentence]]:
    """Swaps a number in the received sentences.

    It searches for numbers in the original records using a regular expression and
    then uses Google's mt5 to replace the one of the found expressions with text.

    It also ensures that the generated sentences are not equal to the original
    sentences and that they have the same number of numbers (by using the same
    regular expression).

    Args:
        sentences: Records to perturb.
        mt5_model: mt5 model to use.
        mt5_tokenizer: mt5 tokenizer to use.
        rng: Numpy random generator to use.
        gpu: Whether to use gpu.

    Returns:
        The perturbed records. Returns None for sentences for which
        the validations failed.
    """

    transformed = swap_number_transform(sentences, mt5_model, mt5_tokenizer, rng, gpu)
    return swap_number_validation(sentences, transformed)


def swap_number_transform(
    sentences: DataLike[SentenceLike],
    mt5_model: transformers.MT5ForConditionalGeneration,
    mt5_tokenizer: transformers.T5Tokenizer,
    rng: np.random.Generator,
    gpu: bool = False,
) -> Data[Sentence]:
    """Performs the transform phase for the swap_number perturbation.

    Args:
        sentences: Records to perturb.
        mt5_model: mt5 model to use.
        mt5_tokenizer: mt5 tokenizer to use.
        rng: Numpy random generator to use.
        gpu: Whether to use gpu.

    Returns:
        Transformed sentences.
    """
    masked = ops.mask_detections(
        sentences,
        detect_func=ops.regex_detect_numbers,
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


def swap_number_validation(
    originals: DataLike[SentenceLike],
    transformed: DataLike[Optional[SentenceLike]],
) -> Data[Optional[Sentence]]:
    """Performs the validation phase for the swap_number perturbation.

    It validates that the generated sentences are different from
    the original, and ensures a basic quality level by removing
    sentences that match the mT5 masking pattern (<extra_id_\\d{1,2}>)
    and sentences with character insertions for <>()[]{}_, as they are
    likely model hallucinations.

    It also validates that the original and transformed sentences have
    the same count of numbers to ensure the mT5 model generated a number.

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
            and ops.equal_numbers_count(o, p)
        )

    return functional.lift_boolean_validation(val_func)(originals, transformed)
