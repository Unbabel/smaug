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
    perturbations: DataLike[Optional[SentenceLike]],
) -> Data[Optional[Sentence]]:
    def val_func(o: Sentence, p: Sentence) -> bool:
        return (
            o != p
            and re.search(r"<extra_id_\d{1,2}>", p.value) is None
            and ops.character_insertions(o, p, "<>()[]{}_") == 0
        )

    return functional.lift_boolean_validation(val_func)(originals, perturbations)
