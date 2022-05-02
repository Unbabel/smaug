import typing

from maug import model
from maug.validation import base


class IsContradiction(base.Validation):
    def __init__(
        self,
        roberta: model.RobertaMNLI,
        original_field: typing.Optional[str] = None,
        critical_field: typing.Optional[str] = None,
    ) -> None:
        super().__init__(
            original_field=original_field,
            critical_field=critical_field,
        )
        self.__roberta = roberta

    def __call__(self, records: typing.List[typing.Dict]) -> typing.List[typing.Dict]:
        for r in records:
            if self.critical_field in r:
                nli_input = (
                    f"{r[self.original_field]} </s></s> {r[self.critical_field]}"
                )
                logits = self.__roberta(nli_input)
                predicted_id = logits.argmax().item()
                if predicted_id != self.__roberta.contradiction_id:
                    del r[self.critical_field]
        return records
