import stanza


def stanza_pos_predict(text: str, use_gpu: bool = False):
    nlp = _load_stanza_pos(use_gpu)
    return nlp(text)


def _load_stanza_pos(use_gpu):
    processors = "tokenize,pos"
    stanza.download(lang="en", processors=processors, logging_level="WARN")
    return stanza.Pipeline(lang="en", processors=processors, use_gpu=use_gpu)
