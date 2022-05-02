import typing

from maug import model
from maug.validation import base


class EqualNamedEntityCount(base.CmpBased):
    """Filters critical records that do NOT have the same named entity count.

    Args:
        ner_model: named entity recognition model to use. Should be configured
            with the correct language.
    """

    def __init__(
        self,
        ner_model: model.StanzaNER,
        original_field: typing.Optional[str] = None,
        critical_field: typing.Optional[str] = None,
    ):
        super().__init__(
            original_field=original_field,
            critical_field=critical_field,
        )
        self.__ner_model = ner_model

    def _verify(
        self,
        original: str,
        critical: str,
    ) -> bool:
        orig_entity_count = len(self.__ner_model(original).entities)
        crit_entity_count = len(self.__ner_model(critical).entities)
        return orig_entity_count == crit_entity_count