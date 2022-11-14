from typing import Dict, List, Optional

from smaug import mask
from smaug import model
from smaug import pipeline
from smaug.transform import base
from smaug.transform import error
from smaug._itertools import repeat_items


class MaskAndFill(base.Transform):
    """Generates critical errors by masking and filling sentences.

    This class generates critical errors based on the following steps:

    Perturbing either the source or the target sentence by masking it and then
    filling the masked tokens.

    The perturbed sentence is verified to be different from the original one.

    Args:
        mask: Mask object to use when masking sentences.
        fill: Model to fill the masked sentences.
        num_samples: Number of generated samples to create.
        critical_field: Field to add inside the perturbations dictionary.
    """

    __NAME = "mask-and-fill"
    __MISTRANSLATION_MASKS = (mask.RandomReplace, mask.NamedEntity)
    __HALLUCINATION_MASKS = (mask.RandomInsert,)

    def __init__(
        self,
        mask: mask.Mask,
        fill: model.MaskedLanguageModel,
        num_samples: int = 1,
        critical_field: Optional[str] = None,
    ):
        super().__init__(
            name=self.__NAME,
            error_type=self.__get_type(mask),
            critical_field=critical_field,
        )
        self.__masking = mask
        self.__fill = fill
        self.__num_samples = num_samples

    def __call__(self, original: List[pipeline.State]) -> List[pipeline.State]:
        repeated_items: List[pipeline.State] = list(
            repeat_items(original, self.__num_samples)
        )

        original_sentences = [x.original for x in repeated_items]
        masked = self.__masking(original_sentences)
        filled = self.__fill(masked)

        for orig, t in zip(repeated_items, filled.text):
            orig.perturbations[self.critical_field] = t

        for orig, s in zip(repeated_items, filled.spans):
            orig.metadata[self.critical_field] = s

        return repeated_items

    def __get_type(self, masking) -> error.ErrorType:
        if isinstance(masking, self.__MISTRANSLATION_MASKS):
            return error.ErrorType.MISTRANSLATION
        elif isinstance(masking, self.__HALLUCINATION_MASKS):
            return error.ErrorType.HALLUCINATION
        return error.ErrorType.UNDEFINED
