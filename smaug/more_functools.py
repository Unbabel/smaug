import functools

from typing import Callable


def pipe(*funcs: Callable) -> Callable:
    """Creates a new function by piping the results of the received functions.

    The functions are piped in order, meaning
    pipe(f, g, h)(x) = h(g(f(x)))

    Args:
        funcs: Functions to pipe.

    Returns:
        Function piping the received functions"""
    return functools.reduce(lambda f, g: lambda x: g(f(x)), funcs)
