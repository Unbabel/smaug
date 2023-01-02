from smaug.core import SpanIndexLike, Modification, ModificationTrace, Sentence
from smaug.promote import promote_to_span_index
from smaug.ops.modification import apply_modification


def insert(s: Sentence, span: str, idx: int) -> Sentence:
    """Creates a new sentence by inserting the span at the given idx.

    Args:
        s: Sentence to modify.
        span: Span to insert.
        idx: Index where to start the span insertion.

    Returns:
        Sentence with the insertion operation applied.
    """
    # An insertion is a replacement of the empty string at position idx
    # by the given span.
    # Set index to int as numpy.int64 is not serializable.

    return modify_sentence(s, Modification(old="", new=span, idx=int(idx)))


def replace(s: Sentence, span: str, loc: SpanIndexLike) -> Sentence:
    """Creates a new sentence by replacing characters by a new span.

    The new sentence will have the characters indexed by loc replaced
    by the new span.

    Args:
        s: Sentence to modify.
        span: Span to insert.
        loc: Indexes to delimit the text to be replaced by the span.

    Returns:
        Sentence with the replacement operation applied.
    """
    loc = promote_to_span_index(loc)
    old = s.value[loc.start : loc.end]
    return modify_sentence(s, Modification(old=old, new=span, idx=loc.start))


def delete(s: Sentence, loc: SpanIndexLike) -> Sentence:
    """Creates a new sentence by deleting the characters indexed by loc.

    Args:
        s: Sentence to modify.
        loc: Indexes to delimit the text to be deleted.

    Returns:
        Sentence with the deletion operation applied.
    """
    loc = promote_to_span_index(loc)
    to_delete = s.value[loc.start : loc.end]
    # A deletion is a replacement of the span indexed by loc with the
    # empty string.
    return modify_sentence(s, Modification(old=to_delete, new="", idx=loc.start))


def prepend(s: Sentence, span: str) -> Sentence:
    return insert(s, span, 0)


def append(s: Sentence, span: str) -> Sentence:
    return insert(s, span, len(s))


def rstrip(s: Sentence) -> Sentence:
    last_space_idx = len(s)

    while last_space_idx > 0 and s.value[last_space_idx - 1] == " ":
        last_space_idx -= 1

    new_s = s
    if last_space_idx != len(s):
        new_s = delete(s, (last_space_idx, len(s)))

    return new_s


def modify_sentence(s: Sentence, m: Modification) -> Sentence:
    """Creates a new sentence by applying a modification to this sentence.

    Args:
        s: Sentence to modify.
        m: Modification to apply.

    Returns:
        The new sentence.
    """
    new_value = apply_modification(m, s.value)
    new_trace = ModificationTrace(m, s.trace)
    return Sentence(value=new_value, trace=new_trace)


def find(s: Sentence, sub: str, start=None, end=None) -> int:
    return s.value.find(sub, start, end)


def startswith(s: Sentence, prefix, start=None, end=None) -> bool:
    return s.value.startswith(prefix, start, end)


def endswith(s: Sentence, suffix, start=None, end=None) -> bool:
    return s.value.endswith(suffix, start, end)
