import stanza


def stanza_ner_load(lang: str = "en", use_gpu: bool = False) -> stanza.Pipeline:
    """Loads a new pipeline for a given language.

    Args:
        lang: Language of the pipeline.
        use_gpu: Specifies if a gpu should be used if available.

    Returns:
        stanza.Pipeline that performs tokenization and named entity
        recognition.
    """
    processors = "tokenize,ner"
    stanza.download(lang, processors=processors, logging_level="WARN")
    return stanza.Pipeline(lang, processors=processors, use_gpu=use_gpu)


def stanza_pos_load(lang: str = "en", use_gpu: bool = False) -> stanza.Pipeline:
    processors = "tokenize,pos"
    stanza.download(lang=lang, processors=processors, logging_level="WARN")
    return stanza.Pipeline(lang=lang, processors=processors, use_gpu=use_gpu)
