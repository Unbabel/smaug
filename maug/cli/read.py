import click
import pandas as pd
import typing

from maug import random
from maug.cli import fmt
from maug.cli import processor


@click.command("read-lines", short_help="Read sentences from a text file.")
@click.option("-p", "--path", required=True, help="Path for file to read.")
@click.option("-l", "--lang", required=True, help="Language for the sentences.")
@click.option(
    "-s",
    "--sample",
    type=int,
    help="Number of sentences to sample. If not specified, all sentences are used.",
)
@processor.make
def read_lines(prev, path: str, lang: str, sample: typing.Union[int, None]):
    """Reads sentences from a text file.

    The file is expected to have one sentence per line. The language must
    be specified to enable language aware transformations.
    """

    with open(path, "r") as fp:
        sentences = [l[:-1] for l in fp.readlines()]

    sentences = fmt.pbar_from_iterable(sentences, f"Read Sentences from {path}")
    records = [{"original": s, "perturbations": {}} for s in sentences]

    if sample is not None and len(records) > sample:
        rng = random.numpy_seeded_rng()
        records = rng.choice(records, sample, replace=False).tolist()

    dataset = {"lang": lang, "records": records}

    stream = [el for el in prev]
    stream.append(dataset)
    return stream


@click.command("read-csv", short_help="Read data from a CSV file.")
@click.option("-p", "--path", required=True, help="Path for file to read.")
@click.option(
    "-s",
    "--sample",
    type=int,
    help="Number of records to sample from each dataset. If not specified, the entire datasets are used.",
)
@processor.make
def read_csv(prev, path, sample: typing.Union[int, None]):
    """Reads records from a csv file.

    The file is expected to have a language pair column "lp", that splits the
    records into multiple datasets, one for each language. Languages are expected
    to be in the format {source_lang}-{target_lang}.
    """

    def df_to_dataset(df: pd.DataFrame, lp: str) -> typing.Dict:
        lp_data = df[df["lp"] == lp]
        records = [
            {"original": row["ref"], "perturbations": {}, **row.to_dict()}
            for _, row in lp_data.iterrows()
        ]
        _, target_lang = lp.split("-")
        return {
            "lang": target_lang,
            "records": records,
        }

    def sample_dataset(ds: typing.Dict):
        not_sampled = ds["records"]
        rng = random.numpy_seeded_rng()
        if len(not_sampled) > sample:
            sampled = rng.choice(not_sampled, sample, replace=False)
            ds["records"] = sampled.tolist()
        return ds

    data = pd.read_csv(path)
    lang_pairs = data["lp"].unique()
    lang_pairs = fmt.pbar_from_iterable(lang_pairs, f"Read CSV from {path}")

    datasets = (df_to_dataset(data, lp) for lp in lang_pairs)
    if sample:
        datasets = (sample_dataset(ds) for ds in datasets)

    stream = [el for el in prev]
    stream.extend(datasets)
    return stream
