import functools
import numpy as np
import stanza
import typing
import torch
import transformers

from typing import Optional, Tuple

from smaug import ops
from smaug.core import Data, DataLike, Sentence, SentenceLike
from smaug.promote import promote_to_data, promote_to_sentence


_PERTURB_TOK = "<|perturb|>"
_BLANK_TOK = "[BLANK]"
_SEP_TOK = "[SEP]"
_EMPTY_TOK = "[EMPTY]"
_ANSWER_TOK = "[ANSWER]"

_NEGATION = "[negation]"


def polyjuice_negate(
    text: DataLike[SentenceLike],
    pos_pipeline: stanza.Pipeline,
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    rng: np.random.Generator,
    cuda: bool = False,
) -> Data[Optional[Sentence]]:
    """Polyjuice model conditioned on negation.

    This model wraps the Polyjuice model presented in the paper
    "Polyjuice: Generating Counterfactuals for Explaining, Evaluating, and Improving Models"
    from Tongshuang Wu, Marco Tulio Ribeiro, Jeffrey Heer, Daniel S. Weld
    at the Association for Computational Linguistics (ACL), 2021.
    The code for this model is available at https://github.com/tongshuangwu/polyjuice.

    This model conditions the previous model for negation, by masking verbs.
    It tries to mask also auxiliary verbs with a given verb.

    POS tagging is performed with the stanza POS tagger.

    Args:
        text: Text input.
        cuda: Whether to usa a cuda enabled gpu or not.

    Returns:
        Negated sentences.
    """
    text = promote_to_data(text)
    sentences = [promote_to_sentence(t) for t in text]

    if cuda:
        model.cuda()

    prompts = [_add_negation_prompt(pos_pipeline, rng, s) for s in sentences]
    with torch.no_grad():
        polyjuice_func = functools.partial(
            _polyjuice_inference,
            tokenizer=tokenizer,
            model=model,
            cuda=cuda,
        )
        outputs = [polyjuice_func(p.value) if p is not None else None for p in prompts]

    return Data(
        _extract_results(p, o) if p is not None else None
        for p, o in zip(prompts, outputs)
    )


def _add_negation_prompt(
    pos_pipeline: stanza.Pipeline, rng: np.random.Generator, sentence: Sentence
) -> Optional[Sentence]:
    tagged = ops.stanza_pos_predict(sentence, pos_pipeline).item()
    possible_mask_intervals = []
    for tagged_sentence in tagged.sentences:
        for i, _ in enumerate(tagged_sentence.words):
            interval = _get_prev_aux_if_verb(tagged_sentence, i)
            if interval:
                possible_mask_intervals.append(interval)
            interval = _get_verb_if_verb(tagged_sentence, i)
            if interval:
                possible_mask_intervals.append(interval)

    if not possible_mask_intervals:
        return None

    mask_start, mask_end = rng.choice(possible_mask_intervals)
    masked = ops.replace(sentence, _BLANK_TOK, (mask_start, mask_end))
    prompt = ops.append(
        ops.prepend(masked, f"{sentence} {_PERTURB_TOK} {_NEGATION} "),
        f" {_SEP_TOK}",
    )

    return prompt


def _polyjuice_inference(
    prompt: str,
    tokenizer: transformers.AutoTokenizer,
    model: transformers.AutoModelForCausalLM,
    cuda: bool,
) -> str:
    inputs = tokenizer(
        prompt,
        padding=False,
        truncation=True,
        max_length=1024 - 1,
        return_tensors="pt",
    )
    if cuda:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        num_beams=5,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id,
        max_length=1024,
        do_sample=False,
        no_repeat_ngram_size=2,
    )

    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]


def _extract_results(
    prompt: Sentence, polyjuice_output: str
) -> typing.Optional[Sentence]:
    prompt_and_answers = polyjuice_output.split(_SEP_TOK)
    if len(prompt_and_answers) < 2:
        return None
    _, answers = prompt_and_answers

    negation_start = ops.find(prompt, _NEGATION)
    negation_end = negation_start + len(_NEGATION)
    # +1 to account for extra space
    prompt_no_prefix = ops.delete(prompt, (0, negation_end + 1))
    sep_start = ops.find(prompt_no_prefix, _SEP_TOK)
    # -1 to account for extra space
    masked_sentence = ops.delete(
        prompt_no_prefix, (sep_start - 1, len(prompt_no_prefix))
    )

    for answer in answers.split(_ANSWER_TOK)[:-1]:
        # Avoid bad escape char by replacing single \ with \\
        answer = answer.strip().replace("\\", "\\\\")
        answer = answer if answer != _EMPTY_TOK else ""
        blank_start = ops.find(masked_sentence, _BLANK_TOK)
        blank_end = blank_start + len(_BLANK_TOK)
        masked_sentence = ops.replace(masked_sentence, answer, (blank_start, blank_end))

    return masked_sentence


def _get_prev_aux_if_verb(sentence, i) -> Optional[Tuple]:
    if sentence.words[i].upos != "VERB" or i == 0:
        return None
    last_aux_idx = i
    while last_aux_idx > 0 and sentence.words[last_aux_idx - 1].upos == "AUX":
        last_aux_idx -= 1
    if last_aux_idx == i:
        return None
    return sentence.words[last_aux_idx].start_char, sentence.words[i].end_char


def _get_verb_if_verb(sentence, i) -> Optional[Tuple]:
    word = sentence.words[i]
    if word.upos != "VERB":
        return None
    return word.start_char, word.end_char
