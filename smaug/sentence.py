import dataclasses

from typing import Optional, Tuple, Union

@dataclasses.dataclass(frozen=True, eq=True, order=True)
class SpanIndex:

    start: int
    end: int

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

    span: str
    loc: SpanIndex

    def apply(self, value: str) -> str:
        return f"{value[:self.loc.start]}{self.span}{value[self.loc.end:]}"


@dataclasses.dataclass(frozen=True)
class Sentence:
    """Represents a sentence that stores applied modifications.
    
    Each sentence stores its parent sentence and the modification
    that was applied to the parent to create this sentence.
    """

    value: str
    
    parent: Optional["Sentence"] = dataclasses.field(default=None)
    
    modification: Optional[Modification] = dataclasses.field(default=None)

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
        loc = SpanIndex(start=idx, end=idx)
        return self.apply_modification(Modification(span=span, loc=loc))

    def delete(self, loc: SpanIndexLike) -> "Sentence":
        """Creates a new sentence by deleting the characters indexed by loc.

        Args:
            loc: Indexes to delimit the text to be deleted.

        Returns:
            Sentence with the deletion operation applied.
        """
        loc = promote_to_span_index(loc)
        # A deletion is a replacement of the span indexed by loc with the
        # empty string.
        return self.apply_modification(Modification(span="", loc=loc))
    
    def replace(self, span: str, loc: SpanIndexLike) -> "Sentence":
        """Creates a new sentence by replacing the characters indexed by loc with the given span.

        The new sentence will have this sentence as a parent.

        Args:
            span: Span to insert.
            loc: Indexes to delimit the text to be replaced by the span.

        Returns:
            Sentence with the replacement operation applied.
        """
        loc = promote_to_span_index(loc)
        return self.apply_modification(Modification(span=span, loc=loc))

    def apply_modification(self, modification: Modification) -> "Sentence":
        """Creates a new sentence by applying a modification to this sentence.

        Args:
            modification: Modification to apply.

        Returns:
            Sentence: The new sentence.
        """
        new_value = modification.apply(self.value)
        return Sentence(value=new_value, parent=self, modification=modification)

    def __str__(self) -> str:
        return self.value
