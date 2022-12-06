import stanza

from smaug import core


def stanza_pos_predict(
    text: core.DataLike[str], pos_pipeline: stanza.Pipeline
) -> core.Data:
    """Predicts part-of-speech tags with stanza POS model."""
    text = core.promote_to_data(text)
    return core.Data(pos_pipeline(t) for t in text)
