import dataclasses
from typing import Any, Callable, Dict, Optional

from smaug import ops
from smaug.core import Data, DataLike, Sentence, SentenceLike
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


def lift_transform(
    func: Callable[[DataLike[SentenceLike]], Data[Optional[Sentence]]],
    perturbation: PerturbationId,
) -> Callable[[DataLike[State]], Data[State]]:
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
