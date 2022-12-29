import numpy as np

from smaug import core
from smaug import pipeline
from smaug import transform


def test_random_delete():
    original = [
        pipeline.State(original="First source sentence with words"),
        pipeline.State(original="Second source sentence to be transformed"),
    ]
    transformed = transform.random_delete(
        original, "critical", np.random.default_rng(), p=0.5
    )
    assert isinstance(transformed, core.Data)
    assert len(original) == len(transformed)

    for o, t in zip(original, transformed):
        assert o.original == t.original

        original_splits = t.original.split()
        critical_splits = t.perturbations["critical"].value.split()

        assert len(critical_splits) <= len(original_splits)
        for word in critical_splits:
            assert word in original_splits
