from typing import Callable, List, TypeVar

_T = TypeVar("_T")
_R = TypeVar("_R")


def batch_func(func: Callable[[_T], _R]) -> Callable[[List[_T]], List[_R]]:
    """Transforms a function that receives one input and produces one output
    to a function that receives a list of inputs and produces a list of outputs."""

    def batched(arg: List[_T]) -> List[_R]:
        return [func(v) for v in arg]

    return batched
