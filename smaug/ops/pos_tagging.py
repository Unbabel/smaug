import stanza

from smaug import core


def stanza_pos_predict(text: core.DataLike[str], use_gpu: bool = False) -> core.Data:
    """Predicts part-of-speech tags with stanza english POS model."""
    text = core.promote_to_data(text)
    nlp = _load_stanza_pos(use_gpu)
    return core.Data(nlp(t) for t in text)


def _load_stanza_pos(use_gpu: bool) -> stanza.Pipeline:
    processors = "tokenize,pos"
    stanza.download(lang="en", processors=processors, logging_level="WARN")
    return stanza.Pipeline(lang="en", processors=processors, use_gpu=use_gpu)
