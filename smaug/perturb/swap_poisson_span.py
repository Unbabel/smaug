import numpy as np
import transformers

from smaug import ops
from smaug.core import Data, DataLike, Sentence, SentenceLike


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
