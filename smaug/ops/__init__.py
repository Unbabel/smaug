"""This packaage specifies the core operations for building pipelines."""

from smaug.ops.modification import (
    apply_modification,
    reverse_modification,
    apply_modification_trace,
    reverse_modification_trace,
    modified_indices_from_trace,
    append_modified_indices,
    compress_modified_indices,
)
from smaug.ops.sentence import (
    insert,
    replace,
    delete,
    prepend,
    append,
    rstrip,
    find,
    startswith,
    endswith,
)
from smaug.ops.pos_tagging import stanza_pos_predict
