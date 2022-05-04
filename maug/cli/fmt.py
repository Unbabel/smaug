import click

from tqdm import tqdm


class TqdmExtraFormat(tqdm):
    """Provides a `total_time` format parameter"""

    @property
    def format_dict(self):
        d = super(TqdmExtraFormat, self).format_dict
        total_time = d["elapsed"] * (d["total"] or 0) / max(d["n"], 1)
        d.update(total_time=self.format_interval(total_time))
        return d


PBAR_FORMAT = (
    "        {percentage:3.0f}% |{bar:40}| [{elapsed}/{total_time}, {rate_inv_fmt}]"
)

DEFAULT_MAX_DESC_LEN = 50


def print_desc(desc):
    click.echo(f"\n{desc}")


def pbar_from_total(total: int, desc: str):
    print_desc(desc)
    return TqdmExtraFormat(total=total, bar_format=PBAR_FORMAT)


def pbar_from_iterable(iterable, desc: str):
    print_desc(desc)
    return TqdmExtraFormat(iterable, bar_format=PBAR_FORMAT)


def no_records_message(desc: str):
    return f"{desc}\n        No records (skipping)."
