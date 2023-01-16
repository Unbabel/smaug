import numpy as np

from smaug import ops
from smaug.core import Data, DataLike, Sentence, SentenceLike
from smaug.promote import promote_to_data, promote_to_sentence


def delete_random_words_transform(
    records: DataLike[SentenceLike],
    rng: np.random.Generator,
    p: float = 0.2,
) -> Data[Sentence]:
    """Deletes random words in the sentences.

    Args:
        records: Records to transform.
        perturbation: Name of the perturbation to consider.
        rng: Numpy generator to use.
        p: Probability of deleting a word.

    Returns:
        Transformed records.
    """

    def next_word_start(s: Sentence, start: int):
        # Try to find next space
        word_delim_idx = ops.find(s, " ", start=start)
        if word_delim_idx == -1:
            # If not space, then we are at the last word
            # and return the remaining sentence.
            word_delim_idx = len(s)
        return word_delim_idx + 1

    def transform(s: SentenceLike) -> Sentence:
        s = promote_to_sentence(s)

        curr_idx = 0
        while curr_idx < len(s):
            word_start_idx = next_word_start(s, curr_idx)
            if rng.random() < p:
                s = ops.delete(s, (curr_idx, word_start_idx))
            else:
                curr_idx = word_start_idx

        return s

    records = promote_to_data(records)
    return Data([transform(s) for s in records])
