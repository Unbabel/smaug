import functools
import logging
import stanza
import typing

from smaug.model import base
from smaug.typing import Text


class StanzaNER(base.TokenClassification):
    """Stanza model for named entity recognition.

    This model supports multiple languages, each with its set of tags. The
    used tags and available (language, tags) pairs are described in
    https://stanfordnlp.github.io/stanza/available_models.html#available-ner-models.

    When a language is specified, the default model for that language is loaded.

    Args:
        use_gpu: Specifies if a gpu should be used if available.
    """

    __FOUR_TAGS = ("PER", "LOC", "ORG", "MISC")

    __EIGHTEEN_TAGS = (
        "PERSON",
        "NORP",  # Nationalities / Religious / Political Group
        "FAC",  # Facility
        "ORG",  # Organization
        "GPE",  # Countries / Cities / States
        "LOC",  # Location
        "PRODUCT",
        "EVENT",
        "WORK_OF_ART",
        "LAW",
        "LANGUAGE",
        "DATE",
        "TIME",
        "PERCENT",
        "MONEY",
        "QUANTITY",
        "ORDINAL",
        "CARDINAL",
    )

    __BLNSP_TAGS = ("EVENT", "LOCATION", "ORGANIZATION", "PERSON", "PRODUCT")

    __ITALIAN_TAGS = ("LOC", "ORG", "PER")

    __MYANMAR_TAGS = (
        "LOC",
        "NE",  # Miscellaneous
        "ORG",
        "PNAME",
        "RACE",
        "TIME",
        "NUM",
    )

    __MODEL_TAGS = {
        "af": __FOUR_TAGS,
        "ar": __FOUR_TAGS,
        "bg": __BLNSP_TAGS,
        "zh": __EIGHTEEN_TAGS,
        "nl": __FOUR_TAGS,
        "en": __EIGHTEEN_TAGS,
        "fi": __FOUR_TAGS,
        "fr": __FOUR_TAGS,
        "de": __FOUR_TAGS,
        "hu": __FOUR_TAGS,
        "it": __ITALIAN_TAGS,
        "my": __MYANMAR_TAGS,
        "ru": __FOUR_TAGS,
        "es": __FOUR_TAGS,
        "uk": __FOUR_TAGS,
        "vi": __FOUR_TAGS,
    }

    def __init__(self, lang: str = "en", use_gpu: bool = False):
        self.__nlp = self.__load_pipeline(lang, use_gpu=use_gpu)
        self.__tags = self.__MODEL_TAGS[lang]

    @property
    def tags(self):
        return self.__tags

    @functools.singledispatchmethod
    def __call__(self, text: Text) -> typing.Any:
        """Performs named entity recognition on the received documents.

        Args:
            text: Sentences to process.

        Returns:
            Documents with the identified named entities.
        """
        pass

    @__call__.register
    def _(self, text: str):
        return self.__nlp(text)

    @__call__.register
    def _(self, text: list):
        return [self.__nlp(t) for t in text]

    @staticmethod
    def is_lang_available(lang: str):
        return lang in StanzaNER.__MODEL_TAGS

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def __load_pipeline(lang, use_gpu):
        """Loads a new pipeline for a given language.

        The pipelines are cached for subsequent loads.

        Args:
            lang: Language of the pipeline.
            use_gpu: Specifies if a gpu should be used if available.

        Returns:
            stanza.Pipeline that performs tokenization and named entity
            recognition.
        """
        log = logging.getLogger(__name__)
        log.info("Loading NER Pipeline for language %s.", lang)
        processors = "tokenize,ner"
        stanza.download(lang, processors=processors, logging_level="WARN")
        return stanza.Pipeline(lang, processors=processors, use_gpu=use_gpu)
