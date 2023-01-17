import numpy as np

from smaug import core
from smaug import perturb


def test_delete_random_words():
    original = core.Data(
        [
            "First source sentence with words",
            "Second source sentence to be transformed",
        ]
    )
    transformed = perturb.delete_random_words_transform(
        original, np.random.default_rng(), p=0.5
    )
    assert isinstance(transformed, core.Data)
    assert len(original) == len(transformed)

    for o, t in zip(original, transformed):
        original_splits = o.split()
        transformed_splits = t.value.split()

        assert len(transformed_splits) <= len(original_splits)
        for word in transformed_splits:
            assert word in original_splits
