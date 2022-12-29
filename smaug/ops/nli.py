import torch
import transformers

from smaug.core import DataLike, SentenceLike
from smaug.promote import promote_to_data, promote_to_sentence


def roberta_mnli_predict(
    text: DataLike[SentenceLike],
    model: transformers.RobertaForSequenceClassification,
    tokenizer: transformers.PreTrainedTokenizerBase,
    cuda: bool = False,
) -> torch.FloatTensor:
    """Performs NLI with RoBERTA on the received sentences.

    Args:
        text: Text inputs to process, in the format premisse </s></s> hypothesis.
        model: RoBERTa model to use.
        tokenizer: RoBERTa tokenizer to use.
        cuda: Whether to use gpu or not.

    Returns:
        Logits for each class.
    """
    text = promote_to_data(text)
    sentences = [promote_to_sentence(t) for t in text]
    if cuda:
        model.cuda()
    with torch.no_grad():
        tokenizer_input = [s.value for s in sentences]
        input_ids = tokenizer(
            tokenizer_input,
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
