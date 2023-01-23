from smaug.broadcast import broadcast_data
from smaug.core import Data, DataLike, Sentence, SentenceLike, Validation
from smaug.promote import promote_to_data, promote_to_sentence

from typing import Callable, Optional


def lift_boolean_validation(
    validation_func: Callable[[Sentence, Sentence], bool]
) -> Validation:
    def validate_single_perturbation(
        o: SentenceLike, p: Optional[SentenceLike]
    ) -> Optional[Sentence]:
        if p is None:
            return None
        o, p = promote_to_sentence(o), promote_to_sentence(p)
        return p if validation_func(o, p) else None

    def validate_all_perturbations(
        originals: DataLike[SentenceLike],
        perturbations: DataLike[Optional[SentenceLike]],
    ) -> Data[Optional[Sentence]]:
        originals = promote_to_data(originals)
        perturbations = promote_to_data(perturbations)
        originals, perturbations = broadcast_data(originals, perturbations)
        return Data(
            [
                validate_single_perturbation(o, p)
                for o, p in zip(originals, perturbations)
            ]
        )

    return validate_all_perturbations
