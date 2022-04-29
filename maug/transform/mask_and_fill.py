import typing

from maug import mask
from maug import model
from maug.transform import base
from maug.transform import error
from maug._itertools import repeat_items


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
    """

    __NAME = "mask-and-fill"
    __MISTRANSLATION_MASKS = (mask.RandomReplace, mask.NamedEntity)
    __HALLUCINATION_MASKS = (mask.RandomInsert,)

    def __init__(
        self,
        mask: mask.Mask,
        fill: model.MaskedLanguageModel,
        num_samples: int = 1,
        original_field: typing.Optional[str] = None,
        critical_field: typing.Optional[str] = None,
    ):
        super().__init__(
            name=self.__NAME,
            error_type=self.__get_type(mask),
            original_field=original_field,
            critical_field=critical_field,
        )
        self.__masking = mask
        self.__fill = fill
        self.__num_samples = num_samples

    def __call__(self, original: typing.List[typing.Dict]) -> typing.List[typing.Dict]:
        repeated_items = list(repeat_items(original, self.__num_samples))

        original = [x[self.original_field] for x in repeated_items]
        masked = self.__masking(original)
        filled = self.__fill(masked)

        return [{self.critical_field: f, **o} for o, f in zip(repeated_items, filled)]

    def __get_type(self, masking) -> error.ErrorType:
        if isinstance(masking, self.__MISTRANSLATION_MASKS):
            return error.ErrorType.MISTRANSLATION
        elif isinstance(masking, self.__HALLUCINATION_MASKS):
            return error.ErrorType.HALLUCINATION
        return error.ErrorType.UNDEFINED
