import functools

from smaug.core import Modification, ModifiedIndices, ModificationTrace, SpanIndex
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


def modified_indices_from_trace(t: ModificationTrace) -> ModifiedIndices:
    return functools.reduce(append_modified_indices, t, ModifiedIndices([]))


def append_modified_indices(
    indices: ModifiedIndices, m: Modification
) -> ModifiedIndices:
    """Merges the indices modified by the modification into these indices.

    Args:
        indices: Indexes to append too.
        m: Modification to merge.

    Returns:
        New modified indices with the previous and merged indices.
    """
    new_idxs = set()
    offset = m.new_span_idx.end - m.old_span_idx.end
    for idx in indices:
        # Modified index before the old span. We just add it as this
        # modification did not change it
        if idx < m.old_span_idx.start:
            new_idxs.add(idx)
        # Modified idx after the old span. This modification changed its
        # position by new_end - old_end.
        elif idx >= m.old_span_idx.end:
            new_idxs.add(idx + offset)
        # Modified idx inside old span. This means we are applying a modification
        # on top of another modification. No need to do anything as all new
        # modified indexes will be added.
        else:
            pass
    new_idxs.update(range(m.new_span_idx.start, m.new_span_idx.end))
    return ModifiedIndices(new_idxs)


def compress_modified_indices(indices: ModifiedIndices) -> frozenlist[SpanIndex]:
    """Compresses adjacent indices into spans.

    Returns:
        Spans of adjacent indices.
    """

    def compress_or_add_new_span(
        spans: frozenlist[SpanIndex], idx: int
    ) -> frozenlist[SpanIndex]:
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

    return functools.reduce(compress_or_add_new_span, indices, frozenlist())
