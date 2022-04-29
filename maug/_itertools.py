import itertools
import typing


from collections import abc


def take(iterable: typing.Iterable, n: int) -> typing.List:
    """Return first n items of the iterable as a list.

    Based on the method in itertools recipes in
    https://docs.python.org/3/library/itertools.html
    """
    return list(itertools.islice(iterable, n))


def repeat_items(iterable: typing.Iterable, n: int) -> typing.Iterable:
    """Repeats each item in an iterable n times.

    This function transforms ['A', 'B', 'C'] (n=2) -> ['A', 'A', 'B', 'B', 'C', 'C']
    """
    repeated_iterables = map(lambda x: itertools.repeat(x, n), iterable)
    return itertools.chain.from_iterable(repeated_iterables)


class ResetableIterator(abc.Iterator):
    """Specifies an iterator that can be reset.

    When the iterator is reset, it should go back to the first element.
    """

    def reset():
        ...
