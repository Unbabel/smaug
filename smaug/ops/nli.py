import torch
import transformers

from smaug import core


def roberta_mnli_predict(
    text: core.DataLike[str],
    model: transformers.AutoModelForSequenceClassification,
    tokenizer: transformers.AutoTokenizer,
    cuda: bool = False,
) -> torch.FloatTensor:
    text = core.promote_to_data(text)

    if cuda:
        model.cuda()
    with torch.no_grad():
        tokenizer_input = [el for el in text]
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
    model: transformers.AutoModelForSequenceClassification,
) -> int:
    return model.config.label2id["CONTRADICTION"]
