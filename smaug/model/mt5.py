import functools
import logging
import re
import transformers

from typing import List

from smaug import _itertools
from smaug.model import base
from smaug.model.typing import MaskingPattern
from smaug.typing import Text


_MASK_REGEX = re.compile(r"<extra_id_\d{1,2}>")


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
        clean_outputs: If replacing output, specifies whether small transformations should
            be aplied to the output sentences to improve their quality.
        cuda: Whether to use cuda enabled gpu or not.
    """

    def __init__(
        self,
        replace_outputs: bool = True,
        clean_outputs: bool = True,
        cuda: bool = False,
    ) -> None:
        super().__init__()
        self._model, self._tokenizer = self.__load()
        if cuda:
            self._model.cuda()
        self._replace_outputs = replace_outputs
        self._clean_outputs = clean_outputs
        self._cuda = cuda

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
        input_ids = self._tokenizer(text, padding=True, return_tensors="pt").input_ids
        if self._cuda:
            input_ids = input_ids.cuda()

        output_ids = self._model.generate(
            input_ids,
            do_sample=True,
            top_k=50,
            num_return_sequences=num_return_sequences,
        )

        outputs = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        if self._replace_outputs:
            outputs = [self.replace_masks(s, o) for s, o in zip(text, outputs)]

            if self._clean_outputs:
                outputs = [self._clean_output(o) for o in (outputs)]

        return outputs

    def replace_masks(self, source, output):
        spans = _MASK_REGEX.split(output)[1:]

        masking_pattern = self.masking_pattern()
        for span in spans:
            mask = next(masking_pattern)
            # Avoid bad escape char by replacing single \ with \\
            source = re.sub(mask, span.strip().replace("\\", "\\\\"), source)

        return source

    def _clean_output(self, output: str) -> str:
        if output.startswith((".", ",", "!", "?")):
            output = output[1:]
        return output.strip()

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def __load():
        log = logging.getLogger(__name__)
        name = "google/mt5-large"
        log.info("Loading %s model.", name)
        model = transformers.MT5ForConditionalGeneration.from_pretrained(name)
        tokenizer = transformers.T5Tokenizer.from_pretrained(name)
        return model, tokenizer
