import stanza

from smaug import core
from smaug import sentence


def stanza_pos_predict(
    text: core.DataLike[sentence.SentenceLike], pos_pipeline: stanza.Pipeline
) -> core.Data:
    """Predicts part-of-speech tags with stanza POS model."""
    text = core.promote_to_data(text)
    sentences = [sentence.promote_to_sentence(t) for t in text]
    return core.Data(pos_pipeline(s.value) for s in sentences)
