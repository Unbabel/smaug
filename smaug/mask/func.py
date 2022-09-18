import io
import numpy as np
import functools
import pandas as pd

from typing import List

from smaug.mask.typing import MaskingIntervals
from smaug.typing import Text
from smaug.model import MaskingPattern
from smaug.mask.base import MaskIterator
from smaug._itertools import ResetableIterator


@functools.singledispatch
def mask(docs: Text, intervals: Text, pattern: MaskingPattern):
    """Masks a sentence according to intervals.

    Mask the given sentence according to the specified intervals. The characters
    in the specified intervals are replaced by the mask token.

    Args:
        docs: documents to mask. Can either be a string or a list of strings.
            If docs is a list of strings, then intervals should have a list of
            intervals for each doc.
        intervals: intervals to mask. Each interval should specify this
            start:end to index the sentence. If docs is a list of strings, then
            intervals should have a list of intervals for each doc.
        pattern: masking pattern to mask the intervals.

    Returns:
        A masked document or a list of masked documents according to the given
        intervals.

    Raises:
        NotImplementedError: The docs type is not list, pandas.Series or str.
    """
    raise NotImplementedError(f"Unsupported docs type: {type(docs)}")


@mask.register(list)
@mask.register(pd.Series)
def _(docs, intervals_list: List[MaskingIntervals], pattern: MaskingPattern):
    """See base mask function."""
    assert len(docs) == len(intervals_list)

    verified_list = [__verify_intervals(intervals) for intervals in intervals_list]
    mask_iter = MaskIterator(pattern)
    return [
        __mask_doc(doc, intervals, mask_iter)
        for doc, intervals in zip(docs, verified_list)
    ]


@mask.register(str)
def _(doc: str, intervals: MaskingIntervals, pattern: MaskingPattern):
    """See base mask function."""
    intervals = __verify_intervals(intervals)
    return __mask_doc(doc, intervals, MaskIterator(pattern))


def __verify_intervals(intervals):
    if len(intervals) == 0:
        return intervals

    intervals = np.array(intervals)

    assert len(intervals.shape) == 2, "intervals shape should be [n x 2]"
    assert np.all(
        intervals[:, 0] < intervals[:, 1]
    ), "intervals should have first the start and after the end"
    intervals.sort(axis=0)
    return intervals


def __mask_doc(doc, intervals, mask_iter: MaskIterator):
    if len(intervals) == 0:
        return doc

    mask_iter.reset()
    # Compute the chunks of the sentence that are not masked
    # There are three options:
    # 1 - Before the first index
    # 2 - After the last index
    # 3 - Between the end of an interval and the start of the next interval
    #
    # As an example, the following intervals [(1:3),(5:8),(12:13)] is
    # flattened to [1, 3, 5, 8, 12, 13] and the chunks to keep are
    # [[:1], [3:5], [8:12], [13:]].
    chunks = []
    indexes = intervals.ravel()
    for interval_idx in range(len(indexes)):
        # First option
        if interval_idx == 0:
            chunks.append(doc[: indexes[interval_idx]])
        # Second option
        elif interval_idx == len(indexes) - 1:
            chunks.append(doc[indexes[interval_idx] :])
        # Third option
        elif interval_idx % 2 == 1:
            chunks.append(doc[indexes[interval_idx] : indexes[interval_idx + 1]])

    buffer = io.StringIO()
    for i, c in enumerate(chunks):
        buffer.write(c)
        if i < len(chunks) - 1:
            buffer.write(next(mask_iter))

    return buffer.getvalue()
