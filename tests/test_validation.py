from maug.validation import strings
from maug.validation import numerical


def test_equal_number_count_validation():
    records = [
        {
            "original": "Original sentence with 123 and .234 that should be kept.",
            "perturbations": {
                "critical": "Critical sentence with .3124 and 23,234 to keep."
            },
        },
        {
            "original": "Original sentence with 123 and .234 that not to keep.",
            "perturbations": {
                "critical": "Critical sentence with only .3124 not to keep.",
            },
        },
        {
            "original": "Original sentence with 12,23.42 and 12334 to keep.",
            "perturbations": {
                "critical": "Critical sentence with 42 and 1.23.41 to keep.",
            },
        },
        {
            "original": "Original sentence with 1.232,00 and 12.23 not to keep.",
            "perturbations": {
                "critical": "Critical sentence with only 1.232,00 not to keep.",
            },
        },
    ]
    expected = [
        {
            "original": "Original sentence with 123 and .234 that should be kept.",
            "perturbations": {
                "critical": "Critical sentence with .3124 and 23,234 to keep.",
            },
        },
        {
            "original": "Original sentence with 123 and .234 that not to keep.",
            "perturbations": {},
        },
        {
            "original": "Original sentence with 12,23.42 and 12334 to keep.",
            "perturbations": {
                "critical": "Critical sentence with 42 and 1.23.41 to keep."
            },
        },
        {
            "original": "Original sentence with 1.232,00 and 12.23 not to keep.",
            "perturbations": {},
        },
    ]
    val = numerical.EqualNumbersCount()
    output = val(records)
    assert expected == output


def test_no_regex_match():
    records = [
        {
            "perturbations": {
                "critical": "Critical sentence with <mask_1> and <mask_2> to filter."
            }
        },
        {"perturbations": {"critical": "Critical sentence without masks to keep."}},
        {
            "perturbations": {
                "critical": "Critical sentence with only <mask_3> not to keep."
            }
        },
    ]
    expected = [
        {"perturbations": {}},
        {"perturbations": {"critical": "Critical sentence without masks to keep."}},
        {"perturbations": {}},
    ]
    val = strings.NoRegexMatch(r"<mask_\d+>")
    output = val(records)
    assert expected == output


def test_max_char_insertions():
    records = [
        {
            "original": "Some sentence without any special characters.",
            "perturbations": {
                "critical": "Other sentence without chars.",
            },
        },
        {
            "original": "Some sentence without any special characters.",
            "perturbations": {
                "critical": "Other sentence with <special chars>.",
            },
        },
        {
            "original": "Some sentence with some <special characters.",
            "perturbations": {
                "critical": "Other sentence with mode <special chars].",
            },
        },
        {
            "original": "Sentence with two special characters[].",
            "perturbations": {
                "critical": "Other sentence with many <<<special chars].",
            },
        },
    ]
    expected = [
        {
            "original": "Some sentence without any special characters.",
            "perturbations": {
                "critical": "Other sentence without chars.",
            },
        },
        {
            "original": "Some sentence without any special characters.",
            "perturbations": {},
        },
        {
            "original": "Some sentence with some <special characters.",
            "perturbations": {
                "critical": "Other sentence with mode <special chars].",
            },
        },
        {
            "original": "Sentence with two special characters[].",
            "perturbations": {},
        },
    ]
    val = strings.LeqCharInsertions(chars="<>()[]{}", max_insertions=1)
    validated = val(records)
    assert expected == validated
