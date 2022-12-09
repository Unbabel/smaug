import re

from smaug import pipeline
from smaug import validation


def test_equal_number_count_validation():
    records = [
        pipeline.State(
            original="Original sentence with 123 and .234 that should be kept.",
            perturbations={
                "critical": "Critical sentence with .3124 and 23,234 to keep."
            },
        ),
        pipeline.State(
            original="Original sentence with 123 and .234 that not to keep.",
            perturbations={
                "critical": "Critical sentence with only .3124 not to keep.",
            },
        ),
        pipeline.State(
            original="Original sentence with 12,23.42 and 12334 to keep.",
            perturbations={
                "critical": "Critical sentence with 42 and 1.23.41 to keep.",
            },
        ),
        pipeline.State(
            original="Original sentence with 1.232,00 and 12.23 not to keep.",
            perturbations={
                "critical": "Critical sentence with only 1.232,00 not to keep.",
            },
        ),
    ]
    expected = [
        pipeline.State(
            original="Original sentence with 123 and .234 that should be kept.",
            perturbations={
                "critical": "Critical sentence with .3124 and 23,234 to keep.",
            },
        ),
        pipeline.State(
            original="Original sentence with 123 and .234 that not to keep.",
            perturbations={},
        ),
        pipeline.State(
            original="Original sentence with 12,23.42 and 12334 to keep.",
            perturbations={
                "critical": "Critical sentence with 42 and 1.23.41 to keep."
            },
        ),
        pipeline.State(
            original="Original sentence with 1.232,00 and 12.23 not to keep.",
            perturbations={},
        ),
    ]
    output = validation.equal_numbers_count(records, "critical")
    assert expected == output


def test_no_regex_match():
    records = [
        pipeline.State(
            original="",
            perturbations={
                "critical": "Critical sentence with <mask_1> and <mask_2> to filter."
            },
        ),
        pipeline.State(
            original="",
            perturbations={"critical": "Critical sentence without masks to keep."},
        ),
        pipeline.State(
            original="",
            perturbations={
                "critical": "Critical sentence with only <mask_3> not to keep."
            },
        ),
    ]
    expected = [
        pipeline.State(original="", perturbations={}),
        pipeline.State(
            original="",
            perturbations={"critical": "Critical sentence without masks to keep."},
        ),
        pipeline.State(original="", perturbations={}),
    ]
    output = validation.no_regex_match(records, "critical", re.compile(r"<mask_\d+>"))
    assert expected == output


def test_max_char_insertions():
    records = [
        pipeline.State(
            original="Some sentence without any special characters.",
            perturbations={
                "critical": "Other sentence without chars.",
            },
        ),
        pipeline.State(
            original="Some sentence without any special characters.",
            perturbations={
                "critical": "Other sentence with <special chars>.",
            },
        ),
        pipeline.State(
            original="Some sentence with some <special characters.",
            perturbations={
                "critical": "Other sentence with mode <special chars].",
            },
        ),
        pipeline.State(
            original="Sentence with two special characters[].",
            perturbations={
                "critical": "Other sentence with many <<<special chars].",
            },
        ),
    ]
    expected = [
        pipeline.State(
            original="Some sentence without any special characters.",
            perturbations={
                "critical": "Other sentence without chars.",
            },
        ),
        pipeline.State(
            original="Some sentence without any special characters.",
            perturbations={},
        ),
        pipeline.State(
            original="Some sentence with some <special characters.",
            perturbations={
                "critical": "Other sentence with mode <special chars].",
            },
        ),
        pipeline.State(
            original="Sentence with two special characters[].",
            perturbations={},
        ),
    ]
    validated = validation.leq_char_insertions(
        records, "critical", chars="<>()[]{}", max_insertions=1
    )
    assert expected == validated
