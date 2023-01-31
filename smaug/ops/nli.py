import torch
import transformers

from smaug.broadcast import broadcast_data
from smaug.core import DataLike, SentenceLike
from smaug.promote import promote_to_data


def roberta_mnli_predict(
    premises: DataLike[SentenceLike],
    hypotheses: DataLike[SentenceLike],
    model: transformers.RobertaForSequenceClassification,
    tokenizer: transformers.PreTrainedTokenizerBase,
    cuda: bool = False,
) -> torch.FloatTensor:
    """Performs NLI with RoBERTA on the received sentences.

    Args:
        premises: Premises to process.
        hypotheses: Hypotheses to consider.
        model: RoBERTa model to use.
        tokenizer: RoBERTa tokenizer to use.
        cuda: Whether to use gpu or not.

    Returns:
        Logits for each class.
    """
    premises = promote_to_data(premises)
    hypotheses = promote_to_data(hypotheses)
    premises, hypotheses = broadcast_data(premises, hypotheses)
    inputs = [f"{p} </s></s> {h}" for p, h in zip(premises, hypotheses)]

    if cuda:
        model.cuda()
    with torch.no_grad():
        input_ids = tokenizer(
            inputs,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).input_ids
        if cuda:
            input_ids = input_ids.cuda()
        return model(input_ids).logits


def roberta_mnli_contradiction_id(
    model: transformers.RobertaForSequenceClassification,
) -> int:
    return model.config.label2id["CONTRADICTION"]
