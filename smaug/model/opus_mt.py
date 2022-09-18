import functools
import transformers

from typing import List

from smaug.model import base
from smaug.typing import Text


class OpusMT(base.Text2Text):
    """Opus MT model to perform translation between two languages.

    Args:
        src_lang: language for the received sentences.
        tgt_lang: language for the generated sentences.
        beam_size: parameter for the beam search.
    """

    def __init__(self, src_lang: str, tgt_lang: str, beam_size: int = 4):
        model, tokenizer = self.__load_model_and_tokenizer(
            src_lang,
            tgt_lang,
        )
        self.__model = model
        self.__tokenizer = tokenizer
        self.__beam_size = beam_size

    @functools.singledispatchmethod
    def __call__(self, text: Text) -> Text:
        raise NotImplementedError(f"__call__ not implemented for type {type(text)}")

    @__call__.register
    def _(self, text: str):
        return self.__generate(text)[0]

    @__call__.register
    def _(self, text: list):
        return self.__generate(text)

    def __generate(self, text: Text) -> List[str]:
        model_inputs = self.__tokenizer(text, return_tensors="pt", padding=True)

        translated = self.__model.generate(
            **model_inputs,
            num_beams=self.__beam_size,
        )

        return self.__tokenizer.batch_decode(
            translated,
            skip_special_tokens=True,
        )

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def __load_model_and_tokenizer(src_lang, tgt_lang):
        model_name = "Helsinki-NLP/opus-mt-{}-{}".format(src_lang, tgt_lang)
        tokenizer = transformers.MarianTokenizer.from_pretrained(model_name)
        model = transformers.MarianMTModel.from_pretrained(model_name)
        return model, tokenizer
