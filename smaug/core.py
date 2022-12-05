from typing import List, Tuple, TypeVar, Union

from smaug import _itertools
from smaug import frozen

T = TypeVar("T")


class Data(frozen.frozenlist[T]):
    """Represents a batch of data that can be iterated over.

    This object is immutable.
    """

    def item(self) -> T:
        if len(self) != 1:
            raise ValueError(f"item() can only be called for Data of length 1.")
        return self[0]

    def __repr__(self) -> str:
        values = [str(el) for el in self]
        single_line = ", ".join(values)
        if len(single_line) <= 80:
            return f"Data({single_line})"
        lines = "".join(f"\t{v},\n" for v in values)
        return f"Data[\n" f"{lines}" f"]"


ListLike = Union[List[T], frozen.frozenlist[T]]
DataLike = Union[Data[T], ListLike[T], T]


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


def broadcast_data(*values: Data) -> Tuple[Data, ...]:
    """Broadcasts all values to the length of the longest Data object.

    All objects must either have length 1 or the length of the longest object.

    Args:
        *values (Data): values to broadcast.

    Raises:
        ValueError: If the length of some data object is neither the
        target length or 1.

    Returns:
        Tuple[Data, ...]: tuple with the broadcasted values. This tuple
        has one value for each received argument, corresponding to the
        respective broadcasted value.
    """
    tgt_len = max(len(v) for v in values)
    failed = next((v for v in values if len(v) not in (1, tgt_len)), None)
    if failed:
        raise ValueError(
            f"Unable to broadcast Data of length {len(failed)} to length {tgt_len}: "
            f"received length must the same as target length or 1."
        )
    broadcasted_values = (
        v if len(v) == tgt_len else _itertools.repeat_items(v, tgt_len) for v in values
    )
    return tuple(Data(bv) for bv in broadcasted_values)
