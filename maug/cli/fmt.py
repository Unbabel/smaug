import tqdm

PBAR_FORMAT = (
    "{desc}: |{bar:-20}| {percentage:3.0f}% [ellapsed={elapsed}, remaining={remaining}]"
)

DEFAULT_MAX_DESC_LEN = 50


def pbar_from_total(total: int, desc: str):
    formated_desc = pbar_description(desc)
    return tqdm.tqdm(total=total, desc=formated_desc, bar_format=PBAR_FORMAT)


def pbar_from_iterable(iterable, desc: str):
    formated_desc = pbar_description(desc)
    return tqdm.tqdm(iterable, desc=formated_desc, bar_format=PBAR_FORMAT)


def pbar_description(desc: str, max_desc_len: int = DEFAULT_MAX_DESC_LEN):
    if len(desc) > max_desc_len:
        return f"{desc[:max_desc_len-3]}..."
    else:
        return desc.ljust(max_desc_len, " ")


def no_records_message(desc: str):
    return f"{pbar_description(desc)}: No records (skipping)."
