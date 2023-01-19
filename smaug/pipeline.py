import dataclasses
from typing import Any, Callable, Dict, Optional

from smaug import ops
from smaug.core import Data, DataLike, Sentence, SentenceLike, Validation
from smaug.promote import promote_to_data

PerturbationId = str


@dataclasses.dataclass(frozen=True)
class State:
    """Represents the state of the perturbation process."""

    # The original unmodified sentence.
    original: SentenceLike

    # The sentences with the perturbations, identified by their id.
    perturbations: Dict[PerturbationId, SentenceLike] = dataclasses.field(
        default_factory=dict
    )

    # Other metadata that the perturbations can output, identified by their id.
    metadata: Dict[PerturbationId, Any] = dataclasses.field(default_factory=dict)


PipelineOp = Callable[[DataLike[State]], Data[State]]


def lift_transform(
    func: Callable[[DataLike[SentenceLike]], Data[Optional[Sentence]]],
    perturbation: PerturbationId,
) -> PipelineOp:
    def transform(records: DataLike[State]) -> Data[State]:
        records = promote_to_data(records)
        original = Data([r.original for r in records])
        transformed = func(original)
        for orig, t in zip(records, transformed):
            if t is None:
                continue
            orig.perturbations[perturbation] = t.value
            if t.trace is not None:
                orig.metadata[perturbation] = ops.modified_spans_from_trace(t.trace)
        return records

    return transform


def lift_validation(func: Validation, perturbation: PerturbationId) -> PipelineOp:
    def del_perturbation(state: State):
        if perturbation in state.perturbations:
            del state.perturbations[perturbation]
        if perturbation in state.metadata:
            del state.metadata[perturbation]

    def validation(records: DataLike[State]) -> Data[State]:
        records = promote_to_data(records)
        originals = Data([r.original for r in records])
        transformed = Data([r.perturbations.get(perturbation, None) for r in records])
        validated = func(originals, transformed)
        for r, v in zip(records, validated):
            if v is None:
                del_perturbation(r)
        return records

    return validation
