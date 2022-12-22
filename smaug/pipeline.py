import dataclasses
from typing import Any, Dict

from smaug import sentence

PerturbationId = str


@dataclasses.dataclass(frozen=True)
class State:
    """Represents the state of the perturbation process."""

    # The original unmodified sentence.
    original: sentence.SentenceLike

    # The sentences with the perturbations, identified by their id.
    perturbations: Dict[PerturbationId, sentence.SentenceLike] = dataclasses.field(
        default_factory=dict
    )

    # Other metadata that the perturbations can output, identified by their id.
    metadata: Dict[PerturbationId, Any] = dataclasses.field(default_factory=dict)
