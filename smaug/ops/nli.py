import functools
import torch
import transformers

from smaug import core


def roberta_mnli_predict(
    text: core.DataLike[str], cuda: bool = False
) -> torch.FloatTensor:
    text = core.promote_to_data(text)

    model, tokenizer = _roberta_mnli_load()
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


_roberta_mnli_contradiction_id = None


def roberta_mnli_contradiction_id():
    global _roberta_mnli_contradiction_id
    if _roberta_mnli_contradiction_id is None:
        model, _ = _roberta_mnli_load()
        _roberta_mnli_contradiction_id = model.config.label2id["CONTRADICTION"]
    return _roberta_mnli_contradiction_id


@functools.lru_cache(maxsize=1)
def _roberta_mnli_load():
    name = "roberta-large-mnli"
    model = transformers.AutoModelForSequenceClassification.from_pretrained(name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    return model, tokenizer
