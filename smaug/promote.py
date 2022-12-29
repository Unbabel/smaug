from smaug import frozen
from smaug.core import DataLike, Data, Sentence, SentenceLike, SpanIndex, SpanIndexLike

from typing import TypeVar

T = TypeVar("T")


def promote_to_data(value: DataLike[T]) -> Data[T]:
    """Promotes a value to data.

    The following promotion rules are applied:
    * Data objects are returned as is.
    * Iterable objects are iterated and their elements used for the Data object.
    * All other objects are wrapped in a Data object of length 1.

    Args:
        value (DataLike[T]): Value to promote.

    Returns:
        Data[T]: The Data object corresponding to the promoted value.
    """
    if isinstance(value, Data):
        return value
    if isinstance(value, (list, frozen.frozenlist)):
        return Data(value)
    return Data([value])


def promote_to_span_index(s: SpanIndexLike) -> SpanIndex:
    """Promotes a SpanIndexLike object to SpanIndex.

    Args:
        s: Object to promote.

    Returns:
        Promoted object.
    """
    if isinstance(s, SpanIndex):
        return s
    return SpanIndex(s[0], s[1])


def promote_to_sentence(s: SentenceLike) -> Sentence:
    """Promotes a sentence like object to sentence.

    Args:
        s: Object to promote.

    Returns:
        Promoted object.
    """
    if isinstance(s, Sentence):
        return s
    return Sentence(s)
