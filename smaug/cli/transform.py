import click
import functools

from smaug import core
from smaug import models
from smaug import random
from smaug import perturb
from smaug.cli import accelerator
from smaug.cli import pipeline
from smaug.cli import fmt
from smaug.cli import processor
from smaug.cli import validation


_SWAP_NUM_CMD = "transf-swp-num"
_SWAP_NE_CMD = "transf-swp-ne"
_SWAP_POISSON_SPAN_CMD = "transf-swp-poisson-span"
_NEG_CMD = "transf-neg"
_DEL_PUNCT_SPAN_CMD = "transf-del-punct-span"
_INS_TEXT_CMD = "transf-ins-text"


@click.command(_SWAP_NUM_CMD, short_help="Swap a number for text with regex and mT5.")
@click.option(
    "--batch-size",
    default=16,
    show_default=True,
    help="Batch size when processing records.",
)
@click.option("--no-gpu", is_flag=True, help="Disable gpu.")
@processor.make
@processor.post_run(validation.rm_eq, cli_transforms=[_SWAP_NUM_CMD])
@processor.post_run(
    validation.rm_pattern, pattern=r"<extra_id_\d{1,2}>", cli_transforms=[_SWAP_NUM_CMD]
)
@processor.post_run(
    validation.keep_leq_char_ins,
    chars="<>()[]{}_",
    max_insertions=0,
    cli_transforms=[_SWAP_NUM_CMD],
)
@processor.post_run(validation.keep_eq_num_count, cli_transforms=[_SWAP_NUM_CMD])
@click.pass_context
def swap_num(ctx, datasets, batch_size, no_gpu):
    """Swaps a number for text using regex and mT5.

    This operation is a transformation.
    It searches for numbers in the original records using regular expressions and
    then uses Google's mT5 to replace the one of the found expressions with text.

    The generated sentences are not guarantied to have a new number replacing the
    old one, as the model is free to generate any text.

    It is possible to have other validations to better ensure these conditions
    are met.
    """
    total_records = sum(len(datasets["records"]) for datasets in datasets)
    if total_records == 0:
        click.echo(fmt.no_records_message("Swap a Number for Text"))
        return datasets

    ctx.obj.register_transform(_SWAP_NUM_CMD)

    gpu = accelerator.use_gpu(no_gpu)

    rng = random.numpy_seeded_rng()

    model, tokenizer = models.mT5_load()

    transform_func = functools.partial(
        perturb.swap_number_transform,
        mt5_model=model,
        mt5_tokenizer=tokenizer,
        rng=rng,
        gpu=gpu,
    )

    pipeline_func = pipeline.lift_transform(transform_func, _SWAP_NUM_CMD)

    processed = []

    pbar = fmt.pbar_from_total(total_records, "Swap a Number for Text")
    for dataset in datasets:
        old_records = dataset["records"]
        new_records = []

        for i in range(0, len(old_records), batch_size):
            batch = core.Data(old_records[i : i + batch_size])
            records = pipeline_func(batch)
            new_records.extend(records)
            pbar.update(len(batch))

        dataset["records"] = new_records

        processed.append(dataset)
    return processed


@click.command(
    _SWAP_NE_CMD,
    short_help="Swap a named entity for text with named entity recognition and mT5.",
)
@click.option(
    "--batch-size",
    default=16,
    show_default=True,
    help="Batch size when processing records.",
)
@click.option("--no-gpu", is_flag=True, help="Disable gpu.")
@processor.make
@processor.post_run(validation.rm_eq, cli_transforms=[_SWAP_NE_CMD])
@processor.post_run(
    validation.rm_pattern, pattern=r"<extra_id_\d{1,2}>", cli_transforms=[_SWAP_NE_CMD]
)
@processor.post_run(
    validation.keep_leq_char_ins,
    chars="<>()[]{}_",
    max_insertions=0,
    cli_transforms=[_SWAP_NE_CMD],
)
@processor.post_run(validation.keep_eq_ne_count, cli_transforms=[_SWAP_NE_CMD])
@click.pass_context
def swap_ne(ctx, datasets, batch_size, no_gpu):
    """Swaps a single named entity for text using named entity recognition and mT5.

    This operation is a transformation.
    It searches for named entities in the original records using a stanza model and
    then uses Google's mT5 to replace one of the found expressions with text.

    The generated sentences are not guarantied to have a new named entity replacing the
    old one, as the model is free to generate any text.

    It is possible to have other validations to better ensure these conditions
    are met.
    """
    total_records = sum(
        len(dataset["records"])
        for dataset in datasets
        if models.stanza_ner_lang_available(dataset["lang"])
    )
    if total_records == 0:
        click.echo(fmt.no_records_message("Swap a Named Entitiy for Text"))
        return datasets

    ctx.obj.register_transform(_SWAP_NE_CMD)

    gpu = accelerator.use_gpu(no_gpu)
    rng = random.numpy_seeded_rng()

    model, tokenizer = models.mT5_load()

    processed = []

    pbar = fmt.pbar_from_total(total_records, "Swap a Named Entitiy for Text")
    for dataset in datasets:
        lang = dataset["lang"]
        if not models.stanza_ner_lang_available(lang):
            processed.append(dataset)
            continue
        ner_pipeline = models.stanza_ner_load(lang, gpu)

        transform_func = functools.partial(
            perturb.swap_named_entity_transform,
            ner_pipeline=ner_pipeline,
            mt5_model=model,
            mt5_tokenizer=tokenizer,
            rng=rng,
            gpu=gpu,
        )

        pipeline_func = pipeline.lift_transform(transform_func, _SWAP_NE_CMD)

        old_records = dataset["records"]
        new_records = []

        for i in range(0, len(old_records), batch_size):
            batch = core.Data(old_records[i : i + batch_size])
            records = pipeline_func(batch)
            new_records.extend(records)
            pbar.update(len(batch))

        dataset["records"] = new_records

        processed.append(dataset)
    return processed


@click.command(_NEG_CMD, short_help="Negate the sentence with polyjuice.")
@click.option(
    "--batch-size",
    default=16,
    show_default=True,
    help="Batch size when processing records.",
)
@click.option("--no-gpu", is_flag=True, help="Disable gpu.")
@processor.make
@processor.post_run(validation.rm_eq, cli_transforms=[_NEG_CMD])
@processor.post_run(validation.rm_pattern, pattern="EMPTY", cli_transforms=[_NEG_CMD])
@processor.post_run(validation.keep_contradiction, cli_transforms=[_NEG_CMD])
@click.pass_context
def negate(ctx, datasets, batch_size, no_gpu):
    """Negates the received sentences with polyjuice.

    This operation is a transformation.
    It tries to negate the sentence if possible.
    This transformation is only available for to-english datasets.
    """
    total_records = sum(
        len(orig["records"]) for orig in datasets if orig["lang"] == "en"
    )
    if total_records == 0:
        click.echo(fmt.no_records_message("Negate the Sentence"))
        return datasets

    ctx.obj.register_transform(_NEG_CMD)

    gpu = accelerator.use_gpu(no_gpu)

    rng = random.numpy_seeded_rng()
    pos_pipeline = models.stanza_pos_load("en", gpu)
    model, tokenizer = models.polyjuice_load()

    transform_func = functools.partial(
        perturb.negate_transform,
        pos_pipeline=pos_pipeline,
        polyjuice_model=model,
        polyjuice_tokenizer=tokenizer,
        rng=rng,
        gpu=gpu,
    )

    pipeline_func = pipeline.lift_transform(transform_func, _NEG_CMD)

    pbar = fmt.pbar_from_total(total_records, "Negate the Sentence")

    processed = []
    for orig in datasets:
        if orig["lang"] == "en":
            old_records = orig["records"]
            new_records = []

            for i in range(0, len(old_records), batch_size):
                batch = core.Data(old_records[i : i + batch_size])
                records = pipeline_func(batch)
                new_records.extend(records)
                pbar.update(len(batch))

            orig["records"] = new_records

        processed.append(orig)

    return processed


@click.command(
    _DEL_PUNCT_SPAN_CMD, short_help="Removes a span between two punctuation symbols."
)
@click.option(
    "--low",
    "-l",
    type=int,
    default=4,
    help="minimum number of words for a span to be eligible for deletion.",
    show_default=True,
)
@click.option(
    "--high",
    "-h",
    type=int,
    default=10,
    help="maximum number of words for a span to be eligible for deletion.",
    show_default=True,
)
@processor.make
@click.pass_context
def delete_punct_span(ctx, datasets, low, high):
    """Removes a span between two punctuation symbols.

    This operation is a transformation.
    It detects the following symbols: ,.!? , and deletes a span between two of them.
    It also deletes the symbol to the right of the span.
    """
    total_records = sum(len(datasets["records"]) for datasets in datasets)
    if total_records == 0:
        click.echo(fmt.no_records_message(f"Delete a span between punctuation matches."))
        return datasets

    ctx.obj.register_transform(_DEL_PUNCT_SPAN_CMD)
    rng = random.numpy_seeded_rng()
    transform_func = functools.partial(
        perturb.delete_span_between_punctuation_transform,
        rng=rng,
        low=low,
        high=high,
    )

    pipeline_func = pipeline.lift_transform(transform_func, _DEL_PUNCT_SPAN_CMD)

    processed = []

    pbar = fmt.pbar_from_total(
        total_records, f"Delete a span between punctuation matches."
    )
    for dataset in datasets:
        old_records = core.Data(dataset["records"])
        dataset["records"] = pipeline_func(old_records)

        pbar.update(len(old_records))

        processed.append(dataset)
    return processed


@click.command(_INS_TEXT_CMD, short_help="Insert random text with mT5.")
@click.option(
    "--prob",
    "-p",
    default=0.1,
    show_default=True,
    help="Probability of inserting a mask between two tokens",
)
@click.option(
    "--max-masks",
    default=3,
    show_default=True,
    help="Max masks to add for mT5 to fill.",
)
@click.option(
    "--batch-size",
    default=16,
    show_default=True,
    help="Batch size when processing records.",
)
@click.option("--no-gpu", is_flag=True, help="Disable gpu.")
@processor.make
@processor.post_run(validation.rm_eq, cli_transforms=[_INS_TEXT_CMD])
@processor.post_run(
    validation.rm_pattern, pattern=r"<extra_id_\d{1,2}>", cli_transforms=[_INS_TEXT_CMD]
)
@click.pass_context
def insert_text(ctx, datasets, prob, max_masks, batch_size, no_gpu):
    """Randomly inserts text using mT5.

    This operation is a transformation.
    It randomly adds masks between the words of the original sentence and
    takes as the output the sentence with the masks filled with mT5.
    """
    total_records = sum(len(datasets["records"]) for datasets in datasets)
    if total_records == 0:
        click.echo(fmt.no_records_message("Insert Text"))
        return datasets

    ctx.obj.register_transform(_INS_TEXT_CMD)

    gpu = accelerator.use_gpu(no_gpu)
    rng = random.numpy_seeded_rng()

    model, tokenizer = models.mT5_load()

    transform_func = functools.partial(
        perturb.insert_text_span_transform,
        mt5_model=model,
        mt5_tokenizer=tokenizer,
        rng=rng,
        p=prob,
        max_masks=max_masks,
        gpu=gpu,
    )

    pipeline_func = pipeline.lift_transform(transform_func, _INS_TEXT_CMD)

    processed = []

    pbar = fmt.pbar_from_total(total_records, _INS_TEXT_CMD)
    for dataset in datasets:
        old_records = dataset["records"]
        new_records = []

        for i in range(0, len(old_records), batch_size):
            batch = core.Data(old_records[i : i + batch_size])
            records = pipeline_func(batch)
            new_records.extend(records)
            pbar.update(len(batch))

        dataset["records"] = new_records

        processed.append(dataset)
    return processed


@click.command(
    _SWAP_POISSON_SPAN_CMD, short_help="Replace random spans of text with mT5."
)
@click.option(
    "--batch-size",
    default=16,
    show_default=True,
    help="Batch size when processing records.",
)
@click.option("--no-gpu", is_flag=True, help="Disable gpu.")
@processor.make
@processor.post_run(validation.rm_eq, cli_transforms=[_SWAP_POISSON_SPAN_CMD])
@processor.post_run(
    validation.rm_pattern,
    pattern=r"<extra_id_\d{1,2}>",
    cli_transforms=[_SWAP_POISSON_SPAN_CMD],
)
@click.pass_context
def swap_poisson_span(ctx, datasets, batch_size, no_gpu):
    """Randomly replaces text spans with sizes following a Poisson distribution.

    This operation is a transformation.
    It masks a span of text on the original sentence, where the number
    of masked words (can be 0) is given by the Poisson distribution, and
    takes as the output the sentence with the masks filled with mT5.
    """
    total_records = sum(len(datasets["records"]) for datasets in datasets)
    if total_records == 0:
        click.echo(fmt.no_records_message("Insert Text"))
        return datasets

    ctx.obj.register_transform(_SWAP_POISSON_SPAN_CMD)

    gpu = accelerator.use_gpu(no_gpu)
    rng = random.numpy_seeded_rng()

    model, tokenizer = models.mT5_load()

    transform_func = functools.partial(
        perturb.swap_poisson_span_transform,
        mt5_model=model,
        mt5_tokenizer=tokenizer,
        rng=rng,
        gpu=gpu,
    )

    pipeline_func = pipeline.lift_transform(transform_func, _SWAP_POISSON_SPAN_CMD)

    processed = []

    pbar = fmt.pbar_from_total(total_records, _SWAP_POISSON_SPAN_CMD)
    for dataset in datasets:
        old_records = dataset["records"]
        new_records = []

        for i in range(0, len(old_records), batch_size):
            batch = core.Data(old_records[i : i + batch_size])
            records = pipeline_func(batch)
            new_records.extend(records)
            pbar.update(len(batch))

        dataset["records"] = new_records

        processed.append(dataset)
    return processed
