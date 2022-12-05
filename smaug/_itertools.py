import itertools
import typing


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


def unique_everseen(iterable, key=None):
    """List unique elements, preserving order. Remember all elements ever seen.

    unique_everseen('AAAABBBCCDAABBB') --> A B C D
    unique_everseen('ABBCcAD', str.lower) --> A B C D

    Based on the method in itertools recipes in
    https://docs.python.org/3/library/itertools.html
    """

    seen = set()

    if key is None:
        key = lambda x: x

    for element in iterable:
        k = key(element)
        if k not in seen:
            seen.add(k)
            yield element
