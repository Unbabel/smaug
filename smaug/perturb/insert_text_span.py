import numpy as np
import transformers

from smaug import ops
from smaug.core import Data, DataLike, Sentence, SentenceLike


def insert_text_span_transform(
    records: DataLike[SentenceLike],
    mt5_model: transformers.MT5ForConditionalGeneration,
    mt5_tokenizer: transformers.T5Tokenizer,
    rng: np.random.Generator,
    p: float = 0.1,
    max_masks: int = 3,
    gpu: bool = False,
) -> Data[Sentence]:
    masked = ops.mask_random_insert(
        records,
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
