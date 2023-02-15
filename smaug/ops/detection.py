import re
import stanza

from smaug.core import Data, DataLike, Sentence, SentenceLike, SpanIndex
from smaug.frozen import frozenlist
from smaug.promote import promote_to_data, promote_to_sentence

from typing import Iterable, Optional, Tuple


def stanza_detect_named_entities(
    text: DataLike[SentenceLike],
    ner_pipeline: stanza.Pipeline,
    filter_entities: Optional[Iterable[str]] = None,
) -> Data[frozenlist[Tuple[int, int]]]:
    """Detects text spans with named entities using the Stanza NER pipeline.

    Args:
        text: Text to process.
        ner_pipeline: Stanza NER pipeline to apply.
        filter_entities: Entity types to accept.

    Returns:
        Spans of detected named entities.
    """
    text = promote_to_data(text)
    sentences = map(promote_to_sentence, text)

    documents = [ner_pipeline(s.value) for s in sentences]

    def process_document(doc):
        detected_entities = doc.entities
        if filter_entities is not None:
            unique_entities = set(filter_entities)
            detected_entities = [
                ent for ent in detected_entities if ent.type in unique_entities
            ]

        return frozenlist([(ent.start_char, ent.end_char) for ent in detected_entities])

    return Data([process_document(doc) for doc in documents])


_DEFAULT_NUMBERS_REGEX = re.compile(r"[-+]?\.?(\d+[.,])*\d+")


def regex_detect_numbers(
    text: DataLike[SentenceLike],
) -> Data[frozenlist[Tuple[int, int]]]:
    """Detects text spans with numbers according to a regular expression.

    Args:
        text: Text to process.

    Returns:
        Spans of detected matches.
    """
    return regex_detect_matches(text, _DEFAULT_NUMBERS_REGEX)


def regex_detect_matches(
    text: DataLike[SentenceLike],
    regex: re.Pattern,
) -> Data[frozenlist[Tuple[int, int]]]:
    """Detects text spans that match a given regex.

    Args:
        text: Text to process.
        regex: Regular Expression to search.

    Returns:
        Spans of detected matches.
    """
    text = promote_to_data(text)
    sentences = map(promote_to_sentence, text)

    def process_sentence(s: Sentence) -> frozenlist[Tuple[int, int]]:
        matches = regex.finditer(s.value)
        return frozenlist([m.span() for m in matches])

    return Data([process_sentence(s) for s in sentences])

_DEFAULT_PUNCTUATION_REGEX = re.compile(r"[!?.,]+")

def regex_detect_spans_between_punctuation(
    text: DataLike[SentenceLike],
) -> Data[frozenlist[SpanIndex]]:
    """Detects text spans between punctuation marks.

    Args:
        text: Text to process.

    Returns:
        Spans between detected punctuation marks.
    """
    return regex_detect_spans_between_matches(text, _DEFAULT_PUNCTUATION_REGEX)

def regex_detect_spans_between_matches(
    text: DataLike[SentenceLike], regex: re.Pattern,
) -> Data[frozenlist[SpanIndex]]:
    """Detects text spans between matches of a given regex.

    Args:
        text: Text to process.
        regex: Regular Expression to search.

    Returns:
        Spans between detected matches.
    """
    text = promote_to_data(text)
    sentences = map(promote_to_sentence, text)

    def process_sentence(s: Sentence) -> frozenlist[SpanIndex]:
        matches = regex.finditer(s.value)
        spans_delims_idxs = [0] + [m.end() for m in matches] + [len(s)]
        # Transform indexes in iterable with (idx1,idx2), (idx2,idx3), ...
        pairwise = zip(spans_delims_idxs, spans_delims_idxs[1:])
        return frozenlist(SpanIndex(s, e) for s, e in pairwise)
    
    return Data([process_sentence(s) for s in sentences])