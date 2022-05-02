import click
import json

from maug.cli import fmt
from maug.cli import processor


@click.command("write-json", short_help="Write records to a JSON file.")
@click.option("-p", "--path", required=True, help="File path to store the records.")
@click.option(
    "--indent",
    default=2,
    type=int,
    show_default=True,
    help="Number of spaces to indent the file.",
)
@processor.make
def write_json(datasets, path, indent):
    """Writes all records to a JSON file.

    This is an utility operation to store the generated records. It converts the records
    to JSON objects and stores them in a file, with a format for easy reading.

    The records are stored in a non-compressed format that is more user friendly.
    If the objective is to reduce file size, another write format should be used.
    """

    records = []
    total_records = sum(len(dataset["records"]) for dataset in datasets)
    pbar = fmt.pbar_from_total(total_records, f"Write JSON to {path}")
    for dataset in datasets:
        records.extend(dataset["records"])
        pbar.update(len(dataset["records"]))

    with open(path, "w") as fp:
        json.dump(records, fp, ensure_ascii=False, indent=indent)
    return datasets
