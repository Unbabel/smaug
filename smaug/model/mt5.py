import functools
import logging
import re
import transformers

from typing import List, Tuple

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
        clean_outputs: bool = True,
        cuda: bool = False,
    ) -> None:
        super().__init__()
        self._model, self._tokenizer = self.__load()
        if cuda:
            self._model.cuda()
        self._clean_outputs = clean_outputs
        self._cuda = cuda

    @classmethod
    def masking_pattern(cls) -> MaskingPattern:
        return _MT5MaskFunction()

    @functools.singledispatchmethod
    def __call__(self, text: Text) -> base.MaskedLanguageModelOutput:
        raise NotImplementedError(f"Not implemented for type {type(text)}")

    @__call__.register
    def _(self, text: str) -> base.MaskedLanguageModelOutput:
        texts_w_spans = self.__generate([text])
        texts = [x[0] for x in texts_w_spans]
        spans = [x[1] for x in texts_w_spans]
        return base.MaskedLanguageModelOutput(text=texts[0], spans=spans[0])

    @__call__.register
    def _(self, text: list) -> base.MaskedLanguageModelOutput:
        texts_w_spans = self.__generate(text)
        texts = [x[0] for x in texts_w_spans]
        spans = [x[1] for x in texts_w_spans]
        return base.MaskedLanguageModelOutput(text=texts, spans=spans)

    def __generate(
        self,
        text: List[str],
        num_return_sequences: int = 1,
    ) -> List[Tuple[str, base.GeneratedSpans]]:
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

        outputs = [self.replace_masks(s, o) for s, o in zip(text, outputs)]

        if self._clean_outputs:
            outputs = [self._clean_output(o, spans) for o, spans in (outputs)]

        return outputs

    def replace_masks(self, source, output):
        spans = _MASK_REGEX.split(output)[1:]

        masking_pattern = self.masking_pattern()
        generated_spans = []
        for span in spans:
            # Avoid bad escape char by replacing single \ with \\
            escaped_span = span.strip().replace("\\", "\\\\")

            mask = next(masking_pattern)

            pattern_match = re.search(mask, source)
            if pattern_match:
                first_idx = pattern_match.start()
                last_idx = first_idx + len(escaped_span)
                generated_spans.append((first_idx, last_idx))

            source = re.sub(mask, escaped_span, source)

        return source, generated_spans

    def _clean_output(
        self, output: str, spans: base.GeneratedSpans
    ) -> Tuple[str, base.GeneratedSpans]:
        while output.startswith((".", ",", "!", "?", " ")):
            output = output[1:]
            spans = [(s[0] - 1, s[1] - 1) for s in spans]
        clean = output.rstrip()
        if len(spans) > 1:
            # Update last span to be atmost the output size
            last = spans[-1]
            spans[-1] = (last[0], min(last[1], len(clean)))
        return clean, spans

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def __load():
        log = logging.getLogger(__name__)
        name = "google/mt5-large"
        log.info("Loading %s model.", name)
        model = transformers.MT5ForConditionalGeneration.from_pretrained(name)
        tokenizer = transformers.T5Tokenizer.from_pretrained(name)
        return model, tokenizer
