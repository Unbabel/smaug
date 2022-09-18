import functools
import torch
import transformers

from typing import List

from smaug.typing import Text


class RobertaMNLI:
    """RoBERTa model for MNLI.

    Args:
        cuda: Whether to use cuda or not.
    """

    def __init__(self, cuda: bool = False) -> None:
        self.__model, self.__tokenizer = self.__load()
        if cuda:
            self.__model.cuda()
        self.__cuda = cuda

    @property
    def contradiction_id(self):
        return self.__model.config.label2id["CONTRADICTION"]

    @functools.singledispatchmethod
    def __call__(self, text: Text) -> torch.FloatTensor:
        raise NotImplementedError(f"Not implemented for type {type(text)}")

    @__call__.register
    def _(self, text: str) -> torch.FloatTensor:
        return self.__predict([text])

    @__call__.register
    def _(self, text: list) -> torch.FloatTensor:
        return self.__predict(text)

    def __predict(self, text: List[str]) -> List[bool]:
        with torch.no_grad():
            input_ids = self.__tokenizer(
                text, padding=True, return_tensors="pt", truncation=True, max_length=512
            ).input_ids
            if self.__cuda:
                input_ids = input_ids.cuda()
            return self.__model(input_ids).logits

    @staticmethod
    def __load():
        name = "roberta-large-mnli"
        model = transformers.AutoModelForSequenceClassification.from_pretrained(name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        return model, tokenizer
