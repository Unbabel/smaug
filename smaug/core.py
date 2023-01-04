import dataclasses

from typing import Iterator, List, Optional, Tuple, TypeVar, Union

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
        values = [repr(el) for el in self]
        single_line = ", ".join(values)
        if len(single_line) <= 80:
            return f"Data[{single_line}]"
        lines = "".join(f"\t{v},\n" for v in values)
        return f"Data[\n" f"{lines}" f"]"


ListLike = Union[List[T], frozen.frozenlist[T]]
DataLike = Union[Data[T], ListLike[T], T]


@dataclasses.dataclass(frozen=True, eq=True, order=True)
class SpanIndex:

    start: int
    end: int

    def encloses(self, other: "SpanIndex") -> bool:
        """Verifies whether this span totally encloses the other.

        If a span A encloses a span B, then:
        A.start  B.start   B.end    A.end
        ---|--------|--------|--------|---
        """
        return self.start <= other.start <= other.end <= self.end

    def partial_overlaps(self, other: "SpanIndex") -> bool:
        """Verifies whether this span partially overlaps the other.

        If a span A partially overlaps span B, then:
        A.start  B.start   A.end    B.end
        ---|--------|--------|--------|---
        or
        B.start  A.start   B.end    A.end
        ---|--------|--------|--------|---
        """
        return (
            self.start <= other.start <= self.end <= other.end
            or other.start <= self.start <= other.end <= self.end
        )

    def intersects(self, other: "SpanIndex") -> bool:
        return (
            self.encloses(other)
            or other.encloses(self)
            or self.partial_overlaps(other)
            or other.partial_overlaps(self)
        )

    def __post_init__(self):
        if self.start < 0:
            raise ValueError(f"'start' must be positive but is {self.start}.")
        if self.end < 0:
            raise ValueError(f"'end' must be positive but is {self.end}.")
        if self.end < self.start:
            msg = f"'end' must be greater or equal to 'start': start={self.start}, end={self.end}"
            raise ValueError(msg)

    def __str__(self) -> str:
        return f"[{self.start}, {self.end}]"


SpanIndexLike = Union[Tuple[int, int], SpanIndex]


@dataclasses.dataclass(frozen=True, eq=True)
class Modification:
    """Stores a modification that was applied to a given sentence.

    Attributes:
        old: The old span to be replaced by new.
        new: The new span to replace old.
        idx: Position where to start replacing.
    """

    old: str
    new: str
    idx: int

    @property
    def old_span_idx(self) -> SpanIndex:
        return SpanIndex(self.idx, self.idx + len(self.old))

    @property
    def new_span_idx(self) -> SpanIndex:
        return SpanIndex(self.idx, self.idx + len(self.new))


@dataclasses.dataclass(frozen=True)
class ModificationTrace:
    """Stores the trace of multiple modifications in order."""

    curr: Modification
    prev: Optional["ModificationTrace"] = dataclasses.field(default=None)

    @staticmethod
    def from_modifications(*modifications: Modification) -> "ModificationTrace":
        """Constructs a modification trace by considering the modifications in order.

        Args:
            modifications: Modifications to store.

        Raises:
            ValueError: If no modifications were provided.

        Returns:
            The modification trace.
        """
        curr = None
        for m in modifications:
            curr = ModificationTrace(m, curr)
        if curr is None:
            raise ValueError("at least on modification is expected.")
        return curr

    def __iter__(self) -> Iterator[Modification]:
        """Creates an iterator to visit the modifications from oldest to newest.

        Returns:
            The iterator object.
        """

        def _yield_modifications(trace: "ModificationTrace") -> Iterator[Modification]:
            if trace.prev is not None:
                yield from _yield_modifications(trace.prev)
            yield trace.curr

        yield from _yield_modifications(self)


@dataclasses.dataclass(frozen=True)
class Sentence:
    """Represents a sentence that stores applied modifications.

    Each sentence stores its value and the modifications trace
    that were applied to this sentence.
    """

    value: str

    trace: Optional[ModificationTrace] = dataclasses.field(default=None)

    def __iter__(self) -> Iterator[str]:
        return iter(self.value)

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Sentence) and self.value == o.value

    def __len__(self) -> int:
        return len(self.value)

    def __str__(self) -> str:
        return self.value


SentenceLike = Union[str, Sentence]
