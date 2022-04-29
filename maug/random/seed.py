import numpy as np
import transformers
import typing

_SEED: typing.Optional[int] = None


def seed_everything(seed: int):
    """Seeds every random based framework to allow for reproducibility.

    Args:
        seed: Seed value to use.
    """
    global _SEED
    _SEED = seed
    transformers.set_seed(seed)


def numpy_seeded_rng() -> np.random.Generator:
    """Creates a numpy.random.Generator with the specified seed.

    If the seed is not defined, a generator without seed is created.

    Returns:
        The created generator.
    """
    if _SEED is None:
        return np.random.default_rng()
    return np.random.default_rng(_SEED)
