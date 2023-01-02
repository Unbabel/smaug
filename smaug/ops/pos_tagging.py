import stanza

from smaug.core import Data, DataLike, SentenceLike
from smaug.promote import promote_to_data, promote_to_sentence


def stanza_pos_predict(
    text: DataLike[SentenceLike], pos_pipeline: stanza.Pipeline
) -> Data:
    """Predicts part-of-speech tags with stanza POS model."""
    text = promote_to_data(text)
    sentences = [promote_to_sentence(t) for t in text]
    return Data(pos_pipeline(s.value) for s in sentences)
