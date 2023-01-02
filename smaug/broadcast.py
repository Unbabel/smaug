from smaug import _itertools
from smaug.core import Data

from typing import Tuple


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
