import functools

from smaug.core import Modification, ModificationTrace, SpanIndex
from smaug.frozen import frozenlist


def apply_modification(m: Modification, value: str) -> str:
    """Replaces old by new in the given string.

    Args:
        m: Modification to apply.
        value: String to apply the modification.

    Raises:
        ValueError: If the given value does not contain old at the given index.

    Returns:
        A string with the applied modification.
    """
    if not value.startswith(m.old, m.idx):
        raise ValueError(f'str "{value}" does not have "{m.old}" at position {m.idx}.')
    replace_start = m.idx
    replace_end = replace_start + len(m.old)
    return f"{value[:replace_start]}{m.new}{value[replace_end:]}"


def reverse_modification(m: Modification, value: str) -> str:
    """Reverses a modification in the given value.

    This operation performs the modification in the
    reverse direction, by replacing old by new.

    Args:
        m: Modification to reverse.
        value: String to apply the modification.

    Raises:
        ValueError: If the given value does not contain new at the given index.

    Returns:
        A string with the applied modification.
    """
    reverse = Modification(old=m.new, new=m.old, idx=m.idx)
    return apply_modification(reverse, value)


def apply_modification_trace(t: ModificationTrace, value: str) -> str:
    """Applies all modifications in order, from the oldest to the newest.

    Args:
        t: Modification trace to apply.
        value: String to apply the modifications.

    Returns:
        Modified string.
    """
    return functools.reduce(lambda acc, mod: apply_modification(mod, acc), t, value)


def reverse_modification_trace(t: ModificationTrace, value: str) -> str:
    """Applies all modifications in reverse order, from the newest to the oldest.

    Args:
        t: Modification trace to reverse
        value: String to apply the modifications.

    Returns:
        Modified string.
    """
    return functools.reduce(
        lambda acc, mod: reverse_modification(mod, acc), reversed(list(t)), value
    )


def modified_spans_from_trace(t: ModificationTrace) -> frozenlist[SpanIndex]:
    """Computes the spans modified by a trace.

    Args:
        t: Modification trace to process.

    Returns:
        Spans of modified indices. Deletions are represented with the empty span.
    """

    def append_modified_spans(
        spans: frozenlist[SpanIndex], m: Modification
    ) -> frozenlist[SpanIndex]:
        # If the modification is a deletion completely on top of an older
        # modification it should be as if the older modification never existed.
        reverting = m.new == "" and any(old == m.old_span_idx for old in spans)
        if reverting:
            spans = [old for old in spans if old != m.old_span_idx]

        new_spans = []
        offset = m.new_span_idx.end - m.old_span_idx.end
        new_span = m.new_span_idx
        for old in spans:
            # Modification after the old span. The old span is unchanged.
            if old.end < m.old_span_idx.start:
                new_spans.append(old)
            # Modification before the old span. The old span must be shifted.
            elif m.old_span_idx.end < old.start:
                shifted = SpanIndex(old.start + offset, old.end + offset)
                new_spans.append(shifted)
            # Modification intersects the old span. The old span must be merged
            # into the new span.
            else:
                new_start = min(old.start, new_span.start)
                new_end = max(old.end + offset, new_span.end)
                new_span = SpanIndex(new_start, new_end)

        # Only add new span if not reverting
        if not reverting:
            new_spans.append(new_span)

        return frozenlist(sorted(new_spans))

    return functools.reduce(append_modified_spans, t, frozenlist())
