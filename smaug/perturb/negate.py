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
    """Negates a given sentence.

    This perturbation uses a POS tagger to identify verbs and their
    preceding auxiliary verbs and then applies Polyjuice to negate
    one of the detected spans.

    It also runs default validations to ensure both a minimum quality level,
    and that the generated text contradicts the original sentences.

    Args:
        sentences: Sentences to transform.
        pos_pipeline: POS pipeline to detect verbs and auxiliary verbs.
        polyjuice_model: Polyjuice model to use for negation.
        polyjuice_tokenizer: Polyjuice tokenizer to use for negation.
        roberta_model: RoBERTa model to use for NLI.
        roberta_tokenizer: RoBERTa tokenizer to use for NLI.
        rng: Numpy random generator to use.
        gpu: Whether to use gpu.

    Returns:
        Perturbed sentences. Returns None for sentences for which
        the transform or the validations failed.
    """
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
    sentences: DataLike[SentenceLike],
    pos_pipeline: stanza.Pipeline,
    polyjuice_model: transformers.AutoModelForCausalLM,
    polyjuice_tokenizer: transformers.PreTrainedTokenizerBase,
    rng: np.random.Generator,
    gpu: bool = False,
) -> Data[Optional[Sentence]]:
    """Performs the transform phase for the negate perturbation.

    Args:
        sentences: Sentences to transform.
        pos_pipeline: POS pipeline to detect verbs and auxiliary verbs.
        polyjuice_model: Polyjuice model to use for negation.
        polyjuice_tokenizer: Polyjuice tokenizer to use for negation.
        rng: Numpy random generator to use.
        gpu: Whether to use gpu.

    Returns:
        Transformed sentences. Returns None for sentences for which
        the transform failed.
    """
    return ops.polyjuice_negate(
        sentences,
        pos_pipeline=pos_pipeline,
        model=polyjuice_model,
        tokenizer=polyjuice_tokenizer,
        rng=rng,
        cuda=gpu,
    )


def negate_validation(
    originals: DataLike[SentenceLike],
    transformed: DataLike[SentenceLike],
    roberta_model: transformers.RobertaForSequenceClassification,
    roberta_tokenizer: transformers.PreTrainedTokenizerBase,
    gpu: bool = False,
) -> Data[Optional[Sentence]]:
    """Performs the validation phase for the negate transform.

    Args:
        originals: Original sentences.
        transformed: Transformed sentences.
        roberta_model: RoBERTa model to use for NLI.
        roberta_tokenizer: RoBERTa tokenizer to use for NLI.
        gpu: Whether to use gpu.

    Returns:
        Validated sentences. Returns None for sentences for which
        the validations failed.
    """

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

    return functional.lift_boolean_validation(val_func)(originals, transformed)
