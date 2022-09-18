from typing import Dict, List, Optional

from smaug import model
from smaug.validation import base


class IsContradiction(base.Validation):
    """Filters perturbed records that do not contradict the original sentence.

    Args:
        original_field: Field in the original records to transform.
        perturbations_field: Field with the perturbations added by the transforms.
            This field is a dictionary with the transform name as keys and the
            perturbed sentences as values.
        critical_field: Field inside the perturbations dictionary with the perturbation
            to test.
    """

    def __init__(
        self,
        roberta: model.RobertaMNLI,
        original_field: Optional[str] = None,
        perturbations_field: Optional[str] = None,
        critical_field: Optional[str] = None,
    ) -> None:
        super().__init__(
            original_field=original_field,
            perturbations_field=perturbations_field,
            critical_field=critical_field,
        )
        self.__roberta = roberta

    def __call__(self, records: List[Dict]) -> List[Dict]:
        for r in records:
            if self.perturbations_field not in r:
                continue
            perturbations = r[self.perturbations_field]
            if self.critical_field not in perturbations:
                continue
            nli_input = f"{r[self.original_field]} </s></s> {perturbations[self.critical_field]}"
            logits = self.__roberta(nli_input)
            predicted_id = logits.argmax().item()
            if predicted_id != self.__roberta.contradiction_id:
                del perturbations[self.critical_field]
        return records
