import dataclasses
import re
import transformers

from typing import List, Tuple

from smaug import core


_MASK_REGEX = re.compile(r"<extra_id_\d{1,2}>")

GeneratedSpan = Tuple[int, int]
GeneratedSpans = List[GeneratedSpan]


@dataclasses.dataclass
class MaskedLanguageModelOutput:

    text: core.Data[str]
    spans: core.Data[GeneratedSpans]


def mT5_generate(
    text: core.DataLike[str],
    model: transformers.MT5ForConditionalGeneration,
    tokenizer: transformers.T5Tokenizer,
    clean_outputs: bool = True,
    cuda: bool = False,
) -> MaskedLanguageModelOutput:
    """Generates with Google's mT5 model.

    Args:
        text: sentences to use as input.
        model: mT5 model to use.
        tokenizer: T5 tokenizer to use.
        clean_outputs: If replacing output, specifies whether small transformations should
            be aplied to the output sentences to improve their quality.
        cuda: Whether to use cuda enabled gpu or not.
    """

    text = core.promote_to_data(text)

    if cuda:
        model.cuda()

    tokenizer_input = [el for el in text]
    input_ids = tokenizer(tokenizer_input, padding=True, return_tensors="pt").input_ids
    if cuda:
        input_ids = input_ids.cuda()

    output_ids = model.generate(
        input_ids,
        max_new_tokens=model.config.max_length,
        do_sample=True,
        top_k=50,
    )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    outputs = [_mT5_replace_masks(s, o) for s, o in zip(text, outputs)]

    if clean_outputs:
        outputs = [_mT5_clean_output(o, spans) for o, spans in (outputs)]

    texts = core.Data(x[0] for x in outputs)
    spans = core.Data(x[1] for x in outputs)

    return MaskedLanguageModelOutput(text=texts, spans=spans)


def mT5_masking_function(idx: int):
    return f"<extra_id_{idx}>"


def _mT5_replace_masks(source: str, output: str) -> Tuple[str, GeneratedSpans]:
    spans = _MASK_REGEX.split(output)[1:]

    generated_spans = []
    mask_idx = 0
    for span in spans:
        # Avoid bad escape char by replacing single \ with \\
        escaped_span = span.strip().replace("\\", "\\\\")

        mask = mT5_masking_function(mask_idx)
        mask_idx += 1

        pattern_match = re.search(mask, source)
        if pattern_match:
            first_idx = pattern_match.start()
            last_idx = first_idx + len(escaped_span)
            generated_spans.append((first_idx, last_idx))

        source = re.sub(mask, escaped_span, source)

    return source, generated_spans


def _mT5_clean_output(output: str, spans: GeneratedSpans) -> Tuple[str, GeneratedSpans]:
    while output.startswith((".", ",", "!", "?", " ")):
        output = output[1:]
        # Can't go below 0 index
        spans = [(s[0] - 1 if s[0] > 0 else 0, s[1] - 1) for s in spans]
    clean = output.rstrip()
    if len(spans) > 1:
        # Update last span to be atmost the output size
        last = spans[-1]
        spans[-1] = (last[0], min(last[1], len(clean)))
    return clean, spans
