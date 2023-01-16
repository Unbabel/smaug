import functools
import numpy as np
import stanza
import transformers

from smaug import ops
from smaug.core import Data, DataLike, Sentence, SentenceLike


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
