"""
This file defines all information required to run the Stanza named entity recognition system.

The documentation for this system is available at 
https://stanfordnlp.github.io/stanza/available_models.html#available-ner-models.
"""
import dataclasses
import functools
import logging
import stanza
import typing

from packaging import version
from smaug.model import base
from smaug.typing import Text
from typing import Tuple


@dataclasses.dataclass
class _StanzaNERModelInfo:

    lang: str
    tags: Tuple[str, ...]
    req_stanza_version: version.Version


# Tags definitions from tag category notes.

_FOUR_TAGS = ("PER", "LOC", "ORG", "MISC")

_EIGHTEEN_TAGS = (
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

_BULGARIAN_BLNSP_TAGS = ("EVENT", "LOCATION", "ORGANIZATION", "PERSON", "PRODUCT")

_FINISH_TURKU_TAGS = ("EVENT", "DATE", "LOC", "ORG", "PER", "PRO")

_ITALIAN_FBK_TAGS = ("LOC", "ORG", "PER")

_JAPANESE_GSD_TAGS = (
    "CARDINAL",
    "DATE",
    "EVENT",
    "FAC",
    "GPE",
    "LANGUAGE",
    "LAW",
    "LOC",
    "MONEY",
    "MOVEMENT",
    "NORP",
    "ORDINAL",
    "ORG",
    "PERCENT",
    "PERSON",
    "PET_NAME",
    "PHONE",
    "PRODUCT",
    "QUANTITY",
    "TIME",
    "TITLE_AFFIX",
    "WORK_OF_ART",
)

# LOC (Location), NE (Misc), ORG (Organization), PNAME (Person)
_MYANMAR_UCSY_TAGS = ("LOC", "NE", "ORG", "PNAME", "RACE", "TIME", "NUM")

_NORWEGIAN_NORNE_TAGS = ("DRV", "EVT", "GPE", "LOC", "MISC", "ORG", "PER", "PROD")

_PERSIAN_ARMAN_TAGS = ("event", "fac", "loc", "org", "pers", "pro")

_SWEDISH_SUC3_TAGS = (
    "animal",
    "inst",
    "myth",
    "person",
    "place",
    "product",
    "other",
    "work",
)

_TURKISH_STARLANG_TAGS = ("LOCATION", "MONEY", "ORGANIZATION", "PERSON", "TIME")

_VIETNAMESE_VLSP_TAGS = ("LOCATION", "MISCELLANEOUS", "ORGANIZATION", "PERSON")

_STANZA_NER_MODEL_INFO = {
    # Afrikaans
    "af": _StanzaNERModelInfo(
        lang="af", tags=_FOUR_TAGS, req_stanza_version=version.Version("1.0.0")
    ),
    # Arabic
    "ar": _StanzaNERModelInfo(
        lang="ar", tags=_FOUR_TAGS, req_stanza_version=version.Version("1.0.0")
    ),
    # Bulgarian
    "bg": _StanzaNERModelInfo(
        lang="bg",
        tags=_BULGARIAN_BLNSP_TAGS,
        req_stanza_version=version.Version("1.2.1"),
    ),
    # Chinese
    "zh": _StanzaNERModelInfo(
        lang="zh", tags=_EIGHTEEN_TAGS, req_stanza_version=version.Version("1.0.0")
    ),
    # Danish
    "da": _StanzaNERModelInfo(
        lang="da", tags=_FOUR_TAGS, req_stanza_version=version.Version("1.4.0")
    ),
    # Dutch
    "nl": _StanzaNERModelInfo(
        lang="nl", tags=_FOUR_TAGS, req_stanza_version=version.Version("1.0.0")
    ),
    # English
    "en": _StanzaNERModelInfo(
        lang="en", tags=_EIGHTEEN_TAGS, req_stanza_version=version.Version("1.0.0")
    ),
    # Finnish
    "fi": _StanzaNERModelInfo(
        lang="fi", tags=_FINISH_TURKU_TAGS, req_stanza_version=version.Version("1.2.1")
    ),
    # French
    "fr": _StanzaNERModelInfo(
        lang="fr", tags=_FOUR_TAGS, req_stanza_version=version.Version("1.0.0")
    ),
    # German
    "de": _StanzaNERModelInfo(
        lang="de", tags=_FOUR_TAGS, req_stanza_version=version.Version("1.0.0")
    ),
    # Hungarian
    "hu": _StanzaNERModelInfo(
        lang="hu", tags=_FOUR_TAGS, req_stanza_version=version.Version("1.2.1")
    ),
    # Italian
    "it": _StanzaNERModelInfo(
        lang="it", tags=_ITALIAN_FBK_TAGS, req_stanza_version=version.Version("1.2.3")
    ),
    # Japanese
    "ja": _StanzaNERModelInfo(
        lang="ja", tags=_JAPANESE_GSD_TAGS, req_stanza_version=version.Version("1.4.0")
    ),
    # Myanmar
    "my": _StanzaNERModelInfo(
        lang="my", tags=_MYANMAR_UCSY_TAGS, req_stanza_version=version.Version("1.4.0")
    ),
    # Norwegian‑Bokmaal
    "nb": _StanzaNERModelInfo(
        lang="nb",
        tags=_NORWEGIAN_NORNE_TAGS,
        req_stanza_version=version.Version("1.4.0"),
    ),
    # Norwegian‑Nynorsk
    "nn": _StanzaNERModelInfo(
        lang="nn",
        tags=_NORWEGIAN_NORNE_TAGS,
        req_stanza_version=version.Version("1.4.0"),
    ),
    # Persian
    "fa": _StanzaNERModelInfo(
        lang="pa", tags=_PERSIAN_ARMAN_TAGS, req_stanza_version=version.Version("1.4.0")
    ),
    # Russian
    "ru": _StanzaNERModelInfo(
        lang="ru", tags=_FOUR_TAGS, req_stanza_version=version.Version("1.0.0")
    ),
    # Spanish
    "es": _StanzaNERModelInfo(
        lang="es", tags=_FOUR_TAGS, req_stanza_version=version.Version("1.0.0")
    ),
    # Swedish
    "sv": _StanzaNERModelInfo(
        lang="sv", tags=_SWEDISH_SUC3_TAGS, req_stanza_version=version.Version("1.4.0")
    ),
    # Turkish
    "tr": _StanzaNERModelInfo(
        lang="tr",
        tags=_TURKISH_STARLANG_TAGS,
        req_stanza_version=version.Version("1.4.0"),
    ),
    # Ukrainian
    "uk": _StanzaNERModelInfo(
        lang="uk", tags=_FOUR_TAGS, req_stanza_version=version.Version("1.0.0")
    ),
    # Vietnamese
    "vi": _StanzaNERModelInfo(
        lang="vi",
        tags=_VIETNAMESE_VLSP_TAGS,
        req_stanza_version=version.Version("1.2.1"),
    ),
}

_AVAILABLE_STANZA_VERSION = version.Version(stanza.__version__)


class StanzaNER(base.TokenClassification):
    """Stanza model for named entity recognition.

    This model supports multiple languages, each with its set of tags. The
    used tags and available (language, tags) pairs are described in
    https://stanfordnlp.github.io/stanza/available_models.html#available-ner-models.

    When a language is specified, the default model for that language is loaded.

    Args:
        use_gpu: Specifies if a gpu should be used if available.
    """

    def __init__(self, lang: str = "en", use_gpu: bool = False):
        self.__nlp = self.__load_pipeline(lang, use_gpu=use_gpu)
        self.__tags = _STANZA_NER_MODEL_INFO[lang].tags

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
        if lang not in _STANZA_NER_MODEL_INFO:
            return False
        model_info = _STANZA_NER_MODEL_INFO[lang]
        if model_info.req_stanza_version > _AVAILABLE_STANZA_VERSION:
            logging.warn(
                'Required Stanza version for language "%s" is "%s" but found "%s".',
                lang,
                model_info.req_stanza_version,
                _AVAILABLE_STANZA_VERSION,
            )
            return False
        return True

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
