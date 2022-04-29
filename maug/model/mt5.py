import functools
import logging
import re
import transformers

from typing import List

from maug import _itertools
from maug.model import base
from maug.model.typing import MaskingPattern
from maug.typing import Text


_MASK_REGEX = re.compile("<extra_id_\d{1,2}>")


class _MT5MaskFunction(_itertools.ResetableIterator):
    def __init__(self) -> None:
        super().__init__()
        self.__counter = 0

    def __next__(self):
        mask = f"<extra_id_{self.__counter}>"
        self.__counter += 1
        return mask

    def reset(self):
        self.__counter = 0


class MT5(base.Text2Text, base.MaskedLanguageModel):
    """Google's mT5 model.

    Args:
        replace_outputs: If using masks in the input, specifies whether the output
            should be the inputs with the replaced masks or the mT5 original masked
            language model output.
        cuda: Whether to use cuda enabled gpu or not.
    """

    def __init__(self, replace_outputs: bool = True, cuda: bool = False) -> None:
        super().__init__()
        self.__model, self.__tokenizer = self.__load()
        if cuda:
            self.__model.cuda()
        self.__replace_outputs = replace_outputs
        self.__cuda = cuda

    @classmethod
    def masking_pattern(cls) -> MaskingPattern:
        return _MT5MaskFunction()

    @functools.singledispatchmethod
    def __call__(self, text: Text) -> Text:
        raise NotImplementedError(f"Not implemented for type {type(text)}")

    @__call__.register
    def _(self, text: str):
        return self.__generate([text])[0]

    @__call__.register
    def _(self, text: list):
        return self.__generate(text)

    def __generate(self, text: List[str], num_return_sequences: int = 1) -> List[str]:
        input_ids = self.__tokenizer(text, padding=True, return_tensors="pt").input_ids
        if self.__cuda:
            input_ids = input_ids.cuda()

        output_ids = self.__model.generate(
            input_ids,
            do_sample=True,
            top_k=50,
            num_return_sequences=num_return_sequences,
        )

        outputs = self.__tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        if self.__replace_outputs:
            outputs = [self.replace_masks(s, o) for s, o in zip(text, outputs)]

        return outputs

    def replace_masks(self, source, output):
        spans = _MASK_REGEX.split(output)[1:]

        masking_pattern = self.masking_pattern()
        for span in spans:
            mask = next(masking_pattern)
            # Avoid bad escape char by replacing single \ with \\
            source = re.sub(mask, span.strip().replace("\\", "\\\\"), source)

        return source

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def __load():
        log = logging.getLogger(__name__)
        name = "google/mt5-large"
        log.info("Loading %s model.", name)
        model = transformers.MT5ForConditionalGeneration.from_pretrained(name)
        tokenizer = transformers.T5Tokenizer.from_pretrained(name)
        return model, tokenizer
