import numpy as np
import stanza
import transformers

from smaug import ops
from smaug.core import Data, DataLike, Sentence, SentenceLike


def negate_transform(
    records: DataLike[SentenceLike],
    pos_pipeline: stanza.Pipeline,
    polyjuice_model: transformers.AutoModelForCausalLM,
    polyjuice_tokenizer: transformers.PreTrainedTokenizerBase,
    rng: np.random.Generator,
    gpu: bool = False,
) -> Data[Sentence]:
    return ops.polyjuice_negate(
        records,
        pos_pipeline=pos_pipeline,
        model=polyjuice_model,
        tokenizer=polyjuice_tokenizer,
        rng=rng,
        cuda=gpu,
    )
