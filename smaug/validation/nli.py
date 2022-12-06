from typing import List, Optional

from smaug import pipeline
from smaug.ops import nli
from smaug.validation import base


class IsContradiction(base.Validation):
    """Filters perturbed records that do not contradict the original sentence.

    Args:
        critical_field: Field inside the perturbations dictionary with the perturbation
            to test.
    """

    def __init__(
        self,
        predict_func,
        contradiction_id: int,
        critical_field: Optional[str] = None,
    ) -> None:
        super().__init__(critical_field=critical_field)
        self._predict_func = predict_func
        self._contradiction_id = contradiction_id

    def __call__(self, records: List[pipeline.State]) -> List[pipeline.State]:
        for r in records:
            if self.critical_field not in r.perturbations:
                continue
            nli_input = f"{r.original} </s></s> {r.perturbations[self.critical_field]}"
            logits = self._predict_func(nli_input)
            predicted_id = logits.argmax().item()
            if predicted_id != self._contradiction_id:
                base.del_perturbation(self.critical_field, r)
        return records
