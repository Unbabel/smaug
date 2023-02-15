import numpy as np

from smaug import ops
from smaug.core import Data, DataLike, Sentence, SentenceLike
from smaug.promote import promote_to_data, promote_to_sentence


from typing import Optional


def delete_span_between_punctuation_transform(
    sentences: DataLike[SentenceLike],
    rng: np.random.Generator,
    low: int = 4,
    high: int = 10,
) -> Data[Optional[Sentence]]:
    """Deletes a text span between two punctuation symbols.

    Args:
        sentences: Sentences to transform.
        rng: Numpy random number generator to use.
        low: Minimum number of words for considered span.
        high: Maximum number of words for considered spans.
    """

    def delete_span(s: Sentence, possible_spans_idxs) -> Optional[Sentence]:
        possible_spans_idxs = [
            span_idx
            for span_idx in possible_spans_idxs
            if span_idx.start > 0 and low < len(s.value[span_idx.start:span_idx.end].split()) < high
        ]
        if len(possible_spans_idxs) == 0:
            return None

        idx_to_drop = rng.choice(possible_spans_idxs)

        return ops.delete(s, idx_to_drop)

    def clean_sentence(s: Sentence) -> Sentence:
        s = ops.rstrip(s)
        # To increase credibility of generated sentence,
        # replace last "," with "." .
        if not ops.endswith(s, (".", "?", "!")):
            s = ops.replace(s, ".", (len(s) - 1, len(s)))
        return s

    sentences = promote_to_data(sentences)
    promoted = Data([promote_to_sentence(s) for s in sentences])
    possible_spans_idxs = ops.regex_detect_spans_between_punctuation(promoted)
    deleted = [delete_span(s, p) for s, p in zip(promoted, possible_spans_idxs)]
    return Data([clean_sentence(s) if s is not None else None for s in deleted])
