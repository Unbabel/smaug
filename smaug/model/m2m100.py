import functools
import transformers

from typing import List

from smaug.model import base
from smaug.typing import Text


class M2M100(base.Text2Text):
    """M2M100 Translation model to perform translation between two languages.

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
        self.__src_lang = src_lang
        self.__tgt_lang = tgt_lang

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
        model_inputs = self.__tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        translated = self.__model.generate(
            **model_inputs,
            num_beams=self.__beam_size,
            forced_bos_token_id=self.__tokenizer.get_lang_id(self.__tgt_lang),
        )

        return self.__tokenizer.batch_decode(
            translated,
            skip_special_tokens=True,
        )

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def __load_model_and_tokenizer(src_lang, tgt_lang):
        model_name = "facebook/m2m100_418M"
        tokenizer = transformers.M2M100Tokenizer.from_pretrained(
            model_name,
            src_lang=src_lang,
            target_lang=tgt_lang,
        )
        model = transformers.M2M100ForConditionalGeneration.from_pretrained(
            model_name,
        )
        return model, tokenizer
