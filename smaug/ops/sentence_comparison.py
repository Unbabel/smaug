import collections
import functools
import nltk
import stanza

from smaug.core import Sentence
from smaug.ops import detection


def equal_numbers_count(s1: Sentence, s2: Sentence) -> bool:
    s1_count = len(detection.regex_detect_numbers(s1).item())
    s2_count = len(detection.regex_detect_numbers(s2).item())
    return s1_count == s2_count


def equal_named_entities_count(
    s1: Sentence, s2: Sentence, ner_pipeline: stanza.Pipeline
) -> bool:
    ner_func = functools.partial(
        detection.stanza_detect_named_entities,
        ner_pipeline=ner_pipeline,
    )
    s1_count = len(ner_func(s1).item())
    s2_count = len(ner_func(s2).item())
    return s1_count == s2_count


def character_insertions(original: Sentence, modified: Sentence, chars: str) -> int:
    """Returns the number of times the given characters were inserted.

    Args:
        original: Original sentence to perform comparison.
        modified: Sentence with modifications.
        chars: Characters to consider.

    Returns:
        The number of inserted characters.
    """
    original_counts = collections.Counter(c for c in original if c in chars)
    modified_counts = collections.Counter(c for c in modified if c in chars)
    insertions = modified_counts - original_counts
    return sum(insertions.values())


def edit_distance(s1: Sentence, s2: Sentence, level: str) -> int:
    """Computes the edit distance between two sentences.

    Args:
        s1: First sentence.
        s2: Second sentence.
        level: Level at which to measure the minimum edit distance. Must be "word" or "char".

    Returns:
        Computed edit distance.
    """

    def char_val_func() -> int:
        return nltk.edit_distance(s1.value, s2.value)

    def word_val_func() -> int:
        return nltk.edit_distance(s1.value.split(), s2.value.split())

    levels = ("char", "word")
    if level not in levels:
        raise ValueError(f"Unknown level {level}: must be one of {levels}.")
    cmp_func = char_val_func if level == "char" else word_val_func
    return cmp_func()
