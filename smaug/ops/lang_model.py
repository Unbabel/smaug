import re
import transformers

from smaug import ops
from smaug.core import Data, DataLike, Sentence, SentenceLike
from smaug.promote import promote_to_data, promote_to_sentence


_MASK_REGEX = re.compile(r"<extra_id_\d{1,2}>")


def mT5_generate(
    text: DataLike[SentenceLike],
    model: transformers.MT5ForConditionalGeneration,
    tokenizer: transformers.T5Tokenizer,
    clean_outputs: bool = True,
    cuda: bool = False,
) -> Data[Sentence]:
    """Generates with Google's mT5 model.

    Args:
        text: sentences to use as input.
        model: mT5 model to use.
        tokenizer: T5 tokenizer to use.
        clean_outputs: If replacing output, specifies whether small transformations should
            be applied to the output sentences to improve their quality.
        cuda: Whether to use cuda enabled gpu or not.
    """

    text = promote_to_data(text)
    sentences = Data(promote_to_sentence(t) for t in text)

    if cuda:
        model.cuda()

    tokenizer_input = [s.value for s in sentences]
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

    outputs = [_mT5_replace_masks(s, o) for s, o in zip(sentences, outputs)]

    if clean_outputs:
        outputs = [_mT5_clean_output(o) for o in outputs]

    return Data(outputs)


def mT5_masking_function(idx: int):
    return f"<extra_id_{idx}>"


def _mT5_replace_masks(source: Sentence, output: str) -> Sentence:
    spans = _MASK_REGEX.split(output)[1:]

    mask_idx = 0
    for span in spans:
        no_space_start = len(span) > 0 and span[0] != " "
        # Avoid bad escape char by replacing single \ with \\
        escaped_span = span.strip().replace("\\", "\\\\")

        mask = mT5_masking_function(mask_idx)
        mask_idx += 1

        if pattern_match := re.search(mask, source.value):
            first_idx = pattern_match.start()
            last_idx = first_idx + len(mask)
            # If we are replacing by a span that does not start by a space,
            # and there is a space before the mask then also remove that space
            # (e.g. near <mask> -> nearly instead of near <mask> -> near ly)
            if first_idx != 0 and source.value[first_idx - 1] == " " and no_space_start:
                first_idx -= 1
            replace_span = (first_idx, last_idx)
            source = ops.replace(source, escaped_span, replace_span)

    return source


def _mT5_clean_output(output: Sentence) -> Sentence:
    while ops.startswith(output, (".", ",", "!", "?", " ")):
        output = ops.delete(output, (0, 1))
    return ops.rstrip(output)
