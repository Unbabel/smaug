import functools
import numpy as np
import stanza
import typing
import torch
import transformers

from typing import Optional, Tuple

from smaug import core
from smaug import sentence
from smaug.ops import pos_tagging


_PERTURB_TOK = "<|perturb|>"
_BLANK_TOK = "[BLANK]"
_SEP_TOK = "[SEP]"
_EMPTY_TOK = "[EMPTY]"
_ANSWER_TOK = "[ANSWER]"

_NEGATION = "[negation]"


def polyjuice_negate(
    text: core.DataLike[sentence.SentenceLike],
    pos_pipeline: stanza.Pipeline,
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    rng: np.random.Generator,
    cuda: bool = False,
) -> core.Data[Optional[sentence.Sentence]]:
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
    text = core.promote_to_data(text)
    sentences = [sentence.promote_to_sentence(t) for t in text]

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

    return core.Data(
        _extract_results(p, o) if p is not None else None
        for p, o in zip(prompts, outputs)
    )


def _add_negation_prompt(
    pos_pipeline: stanza.Pipeline, rng: np.random.Generator, sentence: sentence.Sentence
) -> Optional[sentence.Sentence]:
    tagged = pos_tagging.stanza_pos_predict(sentence, pos_pipeline).item()
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
    masked = sentence.replace(_BLANK_TOK, (mask_start, mask_end))
    prompt = masked.prepend(f"{sentence} {_PERTURB_TOK} {_NEGATION} ").append(
        f" {_SEP_TOK}"
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
    prompt: sentence.Sentence, polyjuice_output: str
) -> typing.Optional[sentence.Sentence]:
    prompt_and_answers = polyjuice_output.split(_SEP_TOK)
    if len(prompt_and_answers) < 2:
        return None
    _, answers = prompt_and_answers

    negation_start = prompt.find(_NEGATION)
    negation_end = negation_start + len(_NEGATION)
    # +1 to account for extra space
    prompt_no_prefix = prompt.delete((0, negation_end + 1))
    sep_start = prompt_no_prefix.find(_SEP_TOK)
    # -1 to account for extra space
    masked_sentence = prompt_no_prefix.delete((sep_start - 1, len(prompt_no_prefix)))

    for answer in answers.split(_ANSWER_TOK)[:-1]:
        answer = answer.strip()
        answer = answer if answer != _EMPTY_TOK else ""
        blank_start = masked_sentence.find(_BLANK_TOK)
        blank_end = blank_start + len(_BLANK_TOK)
        masked_sentence = masked_sentence.replace(answer, (blank_start, blank_end))

    return masked_sentence


def _get_prev_aux_if_verb(sentence, i) -> Optional[Tuple]:
    if sentence.words[i].upos != "VERB" or i == 0:
        return None
    last_aux_idx = i
    while last_aux_idx > 0 and sentence.words[last_aux_idx - 1].upos == "AUX":
        last_aux_idx -= 1
    if last_aux_idx == i:
        return None
    return (sentence.words[last_aux_idx].start_char, sentence.words[i].end_char)


def _get_verb_if_verb(sentence, i) -> Optional[Tuple]:
    word = sentence.words[i]
    if word.upos != "VERB":
        return None
    return (word.start_char, word.end_char)
