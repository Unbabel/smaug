import numpy as np
import re
import stanza
import typing
import torch
import transformers

from typing import Optional, Tuple

from smaug import core
from smaug.ops import pos_tagging


_PERTURB_TOK = "<|perturb|>"
_BLANK_TOK = "[BLANK]"
_SEP_TOK = "[SEP]"
_EMPTY_TOK = "[EMPTY]"
_ANSWER_TOK = "[ANSWER]"

_NEGATION = "[negation]"


def polyjuice_negate(
    text: core.DataLike[str], 
    pos_pipeline: stanza.Pipeline, 
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    rng: np.random.Generator, 
    cuda: bool = False
) -> core.Data[Optional[str]]:
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
        text: text input.
        cuda: Whether to usa a cuda enabled gpu or not.

    Returns:
        Negated sentences
    """
    text = core.promote_to_data(text)

    if cuda:
        model.cuda()

    sentences_with_prompts = [(s, _add_negation_prompt(pos_pipeline, rng, s)) for s in text]
    prompts = [p for _, p in sentences_with_prompts if p is not None]
    with torch.no_grad():
        outputs = []
        for p in prompts:
            inputs = tokenizer(
                p,
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

            output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

            outputs.append(output)

    # We have a result for each prompt, but not for each original
    # sentence.
    results = (_extract_results(o) for o in outputs)

    return core.Data(
        # Replace prompt with result if prompt existed
        next(results) if p is not None else None
        for _, p in sentences_with_prompts
    )


def _add_negation_prompt(pos_pipeline: stanza.Pipeline, rng: np.random.Generator, doc: str) -> Optional[str]:
    tagged = pos_tagging.stanza_pos_predict(doc, pos_pipeline).item()
    possible_mask_intervals = []
    for sentence in tagged.sentences:
        for i, _ in enumerate(sentence.words):
            interval = _get_prev_aux_if_verb(sentence, i)
            if interval:
                possible_mask_intervals.append(interval)
            interval = _get_verb_if_verb(sentence, i)
            if interval:
                possible_mask_intervals.append(interval)

    if not possible_mask_intervals:
        return None
    mask_start, mask_end = rng.choice(possible_mask_intervals)
    masked = f"{doc[:mask_start]}{_BLANK_TOK}{doc[mask_end:]}"
    return f"{doc} {_PERTURB_TOK} {_NEGATION} {masked} {_SEP_TOK}"


def _extract_results(single_output) -> typing.Optional[str]:
    prompt_and_answers = single_output.split(_SEP_TOK)
    if len(prompt_and_answers) < 2:
        return None
    prompt, answers = prompt_and_answers
    _, phrase_with_blank = prompt.split(_NEGATION)
    answers = [x.strip() for x in answers.split(_ANSWER_TOK)][:-1]
    answers = [x if x != _EMPTY_TOK else "" for x in answers]
    for a in answers:
        if a == "":
            phrase_with_blank = re.sub(
                r" %s" % re.escape(_BLANK_TOK), a, phrase_with_blank, count=1
            )
        else:
            phrase_with_blank = re.sub(
                r"%s" % re.escape(_BLANK_TOK), a, phrase_with_blank, count=1
            )
    return phrase_with_blank.strip()


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
