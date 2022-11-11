import functools

from typing import Iterable, List, Optional

from smaug import model
from smaug import random
from smaug.mask import base
from smaug.mask import func
from smaug.model import MaskingPattern
from smaug.typing import Text


class NamedEntity(base.Mask):
    """Masks named entities in a given text according to a probability."""

    def __init__(
        self,
        model: model.StanzaNER,
        pattern: MaskingPattern = None,
        entities: Iterable[str] = None,
        p: float = 1,
        max_masks: Optional[int] = None,
    ):
        super(NamedEntity, self).__init__(pattern=pattern)

        if entities is None:
            entities = model.tags
        self.__entities = set(entities)
        for e in entities:
            if e not in model.tags:
                raise ValueError(f"Unknown entity type: {e}")

        self.__p = p
        self.__max = max_masks
        self.__ner_model = model
        self.__rng = random.numpy_seeded_rng()

    @functools.singledispatchmethod
    def __call__(self, text: Text) -> Text:
        raise NotImplementedError(f"Invalid type {type(text)}")

    @__call__.register
    def _(self, text: str) -> str:
        # Nothing to do
        if self.__p == 0:
            return text
        return self.__mask(text)

    @__call__.register
    def _(self, text: list) -> List[str]:
        # Nothing to do
        if self.__p == 0:
            return text
        return [self.__mask(t) for t in text]

    def __mask(self, text: str) -> str:
        text_w_ner = self.__ner_model(text)

        entities = text_w_ner.entities
        if self.__entities is not None:
            entities = filter(
                lambda ent: ent.type in self.__entities, text_w_ner.entities
            )
        if self.__p != 1:
            entities = filter(lambda _: self.__rng.random() <= self.__p, entities)

        if self.__max:
            entities = list(entities)
            if len(entities) > self.__max:
                entities = self.__rng.choice(entities, self.__max, replace=False)

        intervals = map(lambda ent: (ent.start_char, ent.end_char), entities)
        return func.mask(text, list(intervals), self.pattern)
