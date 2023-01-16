import numpy as np
import re

from smaug import ops
from smaug.core import Data, DataLike, Sentence, SentenceLike
from smaug.promote import promote_to_data, promote_to_sentence


from typing import Optional


def delete_span_between_punctuation_transform(
    records: DataLike[SentenceLike],
    rng: np.random.Generator,
    punctuation: str = ",.!?",
    low: int = 4,
    high: int = 10,
) -> Data[Sentence]:
    def transform(s: SentenceLike) -> Optional[Sentence]:
        s = promote_to_sentence(s)

        matches = punct_regex.finditer(s.value)
        spans_delims_idxs = [0] + [m.end() for m in matches] + [len(s)]
        # Transform indexes in iterable with (idx1,idx2), (idx2,idx3), ...
        pairwise = zip(spans_delims_idxs, spans_delims_idxs[1:])

        possible_spans_idxs = [
            (start, end)
            for start, end in pairwise
            if start > 0 and low < len(s.value[start:end].split()) < high
        ]
        if len(possible_spans_idxs) == 0:
            return None

        idx_to_drop = rng.choice(possible_spans_idxs)

        s = ops.rstrip(ops.delete(s, idx_to_drop))

        # To increase credibility of generated sentence,
        # replace last "," with "." .
        if not ops.endswith(s, (".", "?", "!")):
            s = ops.delete(s, (len(s) - 1, len(s)))
            s = ops.append(s, ".")

        return s

    punct_regex = re.compile(f"[{punctuation}]+")
    records = promote_to_data(records)
    return Data([transform(s) for s in records])
