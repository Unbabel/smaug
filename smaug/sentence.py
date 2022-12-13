import dataclasses
import functools

from typing import Iterable, Iterator, Optional, Tuple, Union

from smaug import frozen


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

    def apply(self, value: str) -> str:
        """Replaces old by new in the given string.

        Args:
            value: String to apply the modification.

        Raises:
            ValueError: If the given value does not contain old at the given index.

        Returns:
            A string with the applied modification.
        """
        if not value.startswith(self.old, self.idx):
            raise ValueError(
                f'str "{value}" does not have "{self.old}" at position {self.idx}.'
            )
        replace_start = self.idx
        replace_end = replace_start + len(self.old)
        return f"{value[:replace_start]}{self.new}{value[replace_end:]}"

    def reverse(self, value: str) -> str:
        """Reverses this modification in the given value.

        This operation performs the modification in the
        reverse direction, by replacing old by new.

        Args:
            value: String to apply the modification.

        Raises:
            ValueError: If the given value does not contain new at the given index.

        Returns:
            A string with the applied modification.
        """
        reverse = Modification(old=self.new, new=self.old, idx=self.idx)
        return reverse.apply(value)


class ModifiedIndices:
    def __init__(self, values: Iterable[int]) -> None:
        self._idxs = sorted(set(values))

    def add(self, modification: Modification) -> "ModifiedIndices":
        """Merges the indices modified by the modification into these indices.

        Args:
            modification: Modification to merge.

        Returns:
            New modifed indices with the previous and merged indices.
        """
        new_idxs = set()
        offset = modification.new_span_idx.end - modification.old_span_idx.end
        for idx in self._idxs:
            # Modified index before the old span. We just add it as this
            # modification did not change it
            if idx < modification.old_span_idx.start:
                new_idxs.add(idx)
            # Modified idx after the old span. This modification changed its
            # position by new_end - old_end.
            elif idx >= modification.old_span_idx.end:
                new_idxs.add(idx + offset)
            # Modified idx inside old span. This means we are applying a modification
            # on top of another modification. No need to do anything as all new
            # modified indexes will be added.
            else:
                pass
        new_idxs.update(
            range(modification.new_span_idx.start, modification.new_span_idx.end)
        )
        return ModifiedIndices(new_idxs)

    def compress(self) -> frozen.frozenlist[SpanIndex]:
        """Compresses adjacent indices into spans.

        Returns:
            Spans of adjacent indices.
        """
        def compress_or_add_new_span(
            spans: frozen.frozenlist[SpanIndex],
            idx: int,
        ) -> frozen.frozenlist[SpanIndex]:
            """Updates the last span with the new index if it is adjacent or creates a new span.

            Args:
                spans: Spans to update.
                idx: New index to add.

            Returns:
                Compressed spans.
            """
            if len(spans) == 0 or spans[-1].end != idx:
                return spans.append(SpanIndex(idx, idx + 1))
            last = spans[-1]
            new_last = SpanIndex(last.start, last.end + 1)
            return spans.replace(len(spans) - 1, new_last)

        return functools.reduce(
            compress_or_add_new_span, self._idxs, frozen.frozenlist()
        )

    def __len__(self) -> int:
        return len(self._idxs)

    def __iter__(self) -> Iterator[int]:
        return iter(self._idxs)

    def __str__(self) -> str:
        return f'{{{", ".join(str(i) for i in self._idxs)}}}'


@dataclasses.dataclass(frozen=True)
class ModificationTrace:
    """Stores the trace of multiple modifications in order."""

    curr: Modification
    prev: Optional["ModificationTrace"] = dataclasses.field(default=None)

    @staticmethod
    def from_modifications(*modifications: Modification) -> "ModificationTrace":
        """Constructs a modification trace by considering the modifications in order.

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

    def apply(self, value: str) -> str:
        """Applies all modifications in order, from the oldest to the newest.

        Args:
            value: String to apply the modifications.

        Returns:
            Modified string.
        """
        return functools.reduce(lambda acc, mod: mod.apply(acc), self, value)

    def reverse(self, value: str) -> str:
        """Applies all modifications in reverse order, from the newest to the oldest.

        Args:
            value: String to apply the modifications.

        Returns:
            Modified string.
        """
        return functools.reduce(
            lambda acc, mod: mod.reverse(acc), reversed(list(self)), value
        )

    def modified_indices(self) -> ModifiedIndices:
        return functools.reduce(ModifiedIndices.add, self, ModifiedIndices([]))

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

    def insert(self, span: str, idx: int) -> "Sentence":
        """Creates a new sentence by inserting the span at the given idx.

        Args:
            span: Span to insert.
            idx: Index where to start the span insertion.

        Returns:
            Sentence with the insertion operation applied.
        """
        # An insertion is a replacement of the empty string at position idx
        # by the given span.
        return self.apply_modification(Modification(old="", new=span, idx=idx))

    def delete(self, loc: SpanIndexLike) -> "Sentence":
        """Creates a new sentence by deleting the characters indexed by loc.

        Args:
            loc: Indexes to delimit the text to be deleted.

        Returns:
            Sentence with the deletion operation applied.
        """
        loc = promote_to_span_index(loc)
        to_delete = self.value[loc.start : loc.end]
        # A deletion is a replacement of the span indexed by loc with the
        # empty string.
        return self.apply_modification(
            Modification(old=to_delete, new="", idx=loc.start)
        )

    def replace(self, span: str, loc: SpanIndexLike) -> "Sentence":
        """Creates a new sentence by replacing characters by a new span.

        The new sentence will have this characters indexed by loc replaced
        by the new span.

        Args:
            span: Span to insert.
            loc: Indexes to delimit the text to be replaced by the span.

        Returns:
            Sentence with the replacement operation applied.
        """
        loc = promote_to_span_index(loc)
        old = self.value[loc.start : loc.end]
        return self.apply_modification(Modification(old=old, new=span, idx=loc.start))

    def apply_modification(self, modification: Modification) -> "Sentence":
        """Creates a new sentence by applying a modification to this sentence.

        Args:
            modification: Modification to apply.

        Returns:
            Sentence: The new sentence.
        """
        new_value = modification.apply(self.value)
        new_trace = ModificationTrace(modification, self.trace)
        return Sentence(value=new_value, trace=new_trace)

    def __str__(self) -> str:
        return self.value
