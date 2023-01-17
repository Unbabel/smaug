import functools
import numpy as np
import stanza
import re
import transformers

from smaug import functional
from smaug import ops
from smaug.core import Data, DataLike, Sentence, SentenceLike

from typing import Optional


def negate(
    sentences: DataLike[SentenceLike],
    pos_pipeline: stanza.Pipeline,
    polyjuice_model: transformers.AutoModelForCausalLM,
    polyjuice_tokenizer: transformers.PreTrainedTokenizerBase,
    roberta_model: transformers.RobertaForSequenceClassification,
    roberta_tokenizer: transformers.PreTrainedTokenizerBase,
    rng: np.random.Generator,
    gpu: bool = False,
) -> Data[Optional[Sentence]]:
    transformed = negate_transform(
        sentences,
        pos_pipeline,
        polyjuice_model,
        polyjuice_tokenizer,
        rng,
        gpu,
    )
    return negate_validation(
        sentences, transformed, roberta_model, roberta_tokenizer, gpu
    )


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


def negate_validation(
    originals: DataLike[SentenceLike],
    perturbations: DataLike[SentenceLike],
    roberta_model: transformers.RobertaForSequenceClassification,
    roberta_tokenizer: transformers.PreTrainedTokenizerBase,
    gpu: bool = False,
):
    def val_func(o: Sentence, p: Sentence) -> bool:
        return (
            o != p
            and re.search("EMPTY", p.value) is None
            and roberta_predict_func(f"{o} </s></s> {p}").argmax().item()
            == roberta_contradiction_id
        )

    roberta_predict_func = functools.partial(
        ops.roberta_mnli_predict,
        model=roberta_model,
        tokenizer=roberta_tokenizer,
        cuda=gpu,
    )
    roberta_contradiction_id = ops.roberta_mnli_contradiction_id(roberta_model)

    return functional.lift_boolean_validation(val_func)(originals, perturbations)
