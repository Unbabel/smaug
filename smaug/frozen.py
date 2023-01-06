import collections.abc

from typing import Generic, Callable, Iterator, Optional, SupportsIndex, Tuple, TypeVar

_T = TypeVar("_T")


class frozenlist(collections.abc.Sequence[_T], Generic[_T]):
    """An immutable variant of Python list."""

    def __init__(self, *args, **kwargs) -> None:
        self._list = list(*args, **kwargs)
        self._hash = None

    def append(self, *add: _T) -> "frozenlist[_T]":
        return self._copy_and_apply(lambda new_list: new_list.extend(add))

    def insert(self, index: SupportsIndex, object: _T) -> "frozenlist[_T]":
        return self._copy_and_apply(lambda new_list: new_list.insert(index, object))

    def replace(self, index: SupportsIndex, object: _T) -> "frozenlist[_T]":
        new_list = list(self._list)
        new_list[index] = object
        new_self = type(self)(new_list)
        return new_self

    def pop(self, index: Optional[SupportsIndex] = None) -> Tuple["frozenlist[_T]", _T]:
        if index is None:
            index = -1
        value = self._list[index]
        new_self = self._copy_and_apply(lambda new_list: new_list.pop(index))
        return new_self, value

    def index(
        self,
        value: _T,
        start: Optional[SupportsIndex] = None,
        stop: Optional[SupportsIndex] = None,
    ) -> int:
        if start is None:
            start = 0
        if stop is None:
            stop = len(self._list)
        return self._list.index(value, start, stop)

    def count(self, value: _T) -> int:
        return self._list.count(value)

    def __len__(self) -> int:
        return len(self._list)

    def __iter__(self) -> Iterator[_T]:
        return iter(self._list)

    def __hash__(self) -> int:
        if self._hash is None:
            h = 0
            for v in self._list:
                h ^= hash(v)
            self._hash = h
        return self._hash

    def __getitem__(self, i) -> _T:
        return self._list[i]

    def __setitem__(self, i, o) -> None:
        raise ValueError("frozenlist is immutable")

    def __delitem__(self, i) -> None:
        raise ValueError("frozenlist is immutable")

    def __contains__(self, o: object) -> bool:
        return self._list.__contains__(o)

    def __add__(self, x: "frozenlist[_T]") -> "frozenlist[_T]":
        return self.append(*x)

    def __str__(self) -> str:
        values = [str(el) for el in self]
        single_line = ", ".join(values)
        if len(single_line) <= 80:
            return f"[{single_line}]"
        lines = "".join(f"\t{v},\n" for v in values)
        return f"[\n{lines}]"

    def __repr__(self) -> str:
        values = [repr(el) for el in self]
        single_line = ", ".join(values)
        if len(single_line) <= 80:
            return f"frozenlist([{single_line}])"
        lines = "".join(f"\t{v},\n" for v in values)
        return f"frozenlist([\n{lines}])"

    def __eq__(self, other):
        return isinstance(other, frozenlist) and self._list == other._list

    def _copy_and_apply(self, func: Callable[[list], None]) -> "frozenlist[_T]":
        new_list = list(self._list)
        func(new_list)
        new_self = type(self)(new_list)
        return new_self
