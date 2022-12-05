import numpy as np
import pytest

from smaug import pipeline
from smaug.transform import deletion
from smaug._itertools import repeat_items


@pytest.mark.parametrize(
    "original,num_samples",
    [
        pytest.param(
            [
                pipeline.State(original="First source sentence with words"),
                pipeline.State(original="Second source sentence to be transformed"),
            ],
            1,
            id="1 critical sample",
        ),
        pytest.param(
            [
                pipeline.State(original="First source sentence with words"),
                pipeline.State(original="Second source sentence to be transformed"),
            ],
            10,
            id="10 critical samples",
        ),
    ],
)
def test_random_delete(original, num_samples):
    transform = deletion.RandomDelete(np.random.default_rng(), num_samples=num_samples)

    transformed = transform(original)
    assert num_samples * len(original) == len(transformed)

    original = repeat_items(original, num_samples)
    for o, t in zip(original, transformed):
        assert o.original == t.original

        original_splits = t.original.split()
        critical_splits = t.perturbations["critical"].split()

        assert len(critical_splits) <= len(original_splits)
        for word in critical_splits:
            assert word in original_splits
