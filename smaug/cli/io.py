import click
import json
import pandas as pd
import typing

from smaug import pipeline
from smaug import random
from smaug.cli import fmt
from smaug.cli import param
from smaug.cli import processor


@click.command("io-read-lines", short_help="Read sentences from a text file.")
@click.option("-p", "--path", required=True, help="Path for file to read.")
@click.option("-l", "--lang", required=True, help="Language for the sentences.")
@click.option(
    "-s",
    "--sample",
    type=param.INT_OR_FLOAT,
    help="Number or percentage of sentences to sample. If not specified, all sentences are used.",
)
@processor.make
def read_lines(prev, path: str, lang: str, sample: typing.Union[int, float, None]):
    """Reads sentences from a text file.

    The file is expected to have one sentence per line. The language must
    be specified to enable language aware transformations.
    """

    with open(path, "r") as fp:
        sentences = [l[:-1] for l in fp.readlines()]

    sentences = fmt.pbar_from_iterable(sentences, f"Read Sentences from {path}")
    records = [pipeline.State(original=s) for s in sentences]

    if sample is not None:
        if isinstance(sample, float):
            sample = int(sample * len(records))

        if len(records) > sample:
            rng = random.numpy_seeded_rng()
            records = rng.choice(records, sample, replace=False).tolist()

    dataset = {"lang": lang, "records": records}

    stream = [el for el in prev]
    stream.append(dataset)
    return stream


@click.command("io-read-csv", short_help="Read data from a CSV file.")
@click.option("-p", "--path", required=True, help="Path for file to read.")
@click.option(
    "-s",
    "--sample",
    type=param.INT_OR_FLOAT,
    help="Number or percentage of sentences to sample. If not specified, all sentences are used.",
)
@processor.make
def read_csv(prev, path, sample: typing.Union[int, None]):
    """Reads records from a csv file.

    The first file column will be interpreted as the language and the
    second as the sentence.
    """

    data = pd.read_csv(path, index_col=False, header=None)
    # Handle empty strings
    data[1].fillna("", inplace=True)
    rows = list(data.iterrows())

    datasets = []
    for idx, row in fmt.pbar_from_iterable(rows, f"Read CSV from {path}"):
        lang = row[0]
        sentence = row[1]

        if idx == 0 or rows[idx - 1][1][0] != lang:
            datasets.append({"lang": lang, "records": []})

        # Always use last dataset
        datasets[-1]["records"].append(pipeline.State(original=sentence))

    if sample is not None:
        for dataset in datasets:
            records = dataset["records"]
            if isinstance(sample, float):
                sample = int(sample * len(records))

            if len(records) > sample:
                rng = random.numpy_seeded_rng()
                records = rng.choice(records, sample, replace=False).tolist()
            dataset["records"] = records

    stream = [el for el in prev]
    stream.extend(datasets)
    return stream


@click.command("io-write-json", short_help="Write records to a JSON file.")
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

    class _StateEncoder(json.JSONEncoder):
        def default(self, o: typing.Any) -> typing.Any:
            if isinstance(o, pipeline.State):
                return {
                    "original": o.original,
                    "perturbations": o.perturbations,
                    "metadata": o.metadata,
                }
            return super().default(o)

    records = []
    total_records = sum(len(dataset["records"]) for dataset in datasets)
    pbar = fmt.pbar_from_total(total_records, f"Write JSON to {path}")
    for dataset in datasets:
        records.extend(dataset["records"])
        pbar.update(len(dataset["records"]))

    with open(path, "w") as fp:
        json.dump(records, fp, ensure_ascii=False, indent=indent, cls=_StateEncoder)
    return datasets
