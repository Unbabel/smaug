"""This package specifies the core operations for performing data augmentation."""

from smaug.ops.modification import (
    apply_modification,
    reverse_modification,
    apply_modification_trace,
    reverse_modification_trace,
    modified_spans_from_trace,
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
from smaug.ops.detection import (
    stanza_detect_named_entities,
    regex_detect_matches,
    regex_detect_numbers,
    regex_detect_spans_between_matches,
    regex_detect_spans_between_punctuation,
)
from smaug.ops.pos_tagging import stanza_pos_predict
from smaug.ops.masking import (
    mask_intervals,
    mask_detections,
    mask_random_replace,
    mask_random_insert,
    mask_poisson_spans,
)
from smaug.ops.lang_model import mT5_generate, mT5_masking_function
from smaug.ops.nli import roberta_mnli_predict, roberta_mnli_contradiction_id
from smaug.ops.text_generation import polyjuice_negate
from smaug.ops.sentence_comparison import (
    character_insertions,
    equal_numbers_count,
    equal_named_entities_count,
    edit_distance,
)
