from nltk import edit_distance_align
from typing import List, Tuple, TypeVar


T = TypeVar("T", str, List)


def mismatched_indexes(a: T, b: T) -> Tuple[List[int], List[int]]:
    """Finds mismatched indexes using edit distance.

    Computes an alignment for the sequences using edit distance,
    and keeps the aligned indexes that do not match an equal
    entry in the oposing sequence.
    """
    align = edit_distance_align(a, b)

    idxs_1 = {i for i in range(len(a))}
    idxs_2 = {i for i in range(len(b))}

    # The first alignment is the (0, 0) state so we discard it
    # (see nltk.metrics.edit_distance_align).
    for align_1, align_2 in align[1:]:
        idx_1 = align_1 - 1
        idx_2 = align_2 - 1
        if a[idx_1] == b[idx_2] and idx_1 in idxs_1 and idx_2 in idxs_2:
            idxs_1.remove(idx_1)
            idxs_2.remove(idx_2)

    return sorted(idxs_1), sorted(idxs_2)


def mismatched_slices(a: T, b: T) -> Tuple[List[slice], List[slice]]:
    """Finds mismatched slices using edit distance.

    Computes mismatched indexes using edit distance (see
    smaug.align.mismatched_indexes).
    Compresses mismatched adjacent indexes into slices. If an index has
    no adjecent indexes, a single entry slice is created.
    """

    def compress_adj_idxs(idxs: List[int]) -> List[slice]:
        if len(idxs) == 0:
            return []
        if len(idxs) == 1:
            return [slice(idxs[0], idxs[0] + 1)]

        compressed = []
        last_start = idxs[0]
        prev = idxs[0]
        for idx in idxs[1:]:
            if idx != prev + 1:
                compressed.append(slice(last_start, prev + 1))
                last_start = idx
            prev = idx
        compressed.append(slice(last_start, idx + 1))

        return compressed

    mis_1, mis_2 = mismatched_indexes(a, b)
    comp_1 = compress_adj_idxs(mis_1)
    comp_2 = compress_adj_idxs(mis_2)
    return comp_1, comp_2
