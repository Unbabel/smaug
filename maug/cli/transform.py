import click

from maug import mask
from maug import model
from maug import transform
from maug.cli import accelerator
from maug.cli import fmt
from maug.cli import processor
from maug.cli import validation


@click.command(
    "transf-swp-num", short_help="Swap a number for text with regex and mT5."
)
@click.option(
    "--batch-size",
    default=16,
    show_default=True,
    help="Batch size when processing records.",
)
@click.option("--no-gpu", is_flag=True, help="Disable gpu.")
@processor.make
@processor.post_run(validation.validation_remove_equal)
@processor.post_run(validation.validation_remove_pattern, pattern="<extra_id_\d{1,2}>")
@click.pass_context
def transform_swap_num(ctx, datasets, batch_size, no_gpu):
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
        click.echo(fmt.no_records_message("Swap Number"))
        return datasets

    transf_id = "transf-swap-number"

    ctx.obj.register_transform(transf_id)

    gpu = accelerator.use_gpu(no_gpu)

    mT5 = model.MT5(cuda=gpu)
    m = mask.Number(model.MT5.masking_pattern(), max_mask=1)
    transf = transform.MaskAndFill(
        mask=m,
        fill=mT5,
        num_samples=1,
        original_field="original",
        critical_field=transf_id,
    )

    processed = []

    pbar = fmt.pbar_from_total(total_records, "Swap Number")
    for dataset in datasets:
        old_records = dataset["records"]
        new_records = []

        for i in range(0, len(old_records), batch_size):
            batch = old_records[i : i + batch_size]
            records = transf(batch)
            new_records.extend(records)
            pbar.update(len(batch))

        dataset["records"] = new_records

        processed.append(dataset)
    return processed


@click.command("transf-neg", short_help="Negate the sentence with polyjuice.")
@click.option(
    "--batch-size",
    default=16,
    show_default=True,
    help="Batch size when processing records.",
)
@click.option("--no-gpu", is_flag=True, help="Disable gpu.")
@processor.make
@processor.post_run(validation.validation_remove_equal)
@processor.post_run(validation.validation_remove_pattern, pattern="EMPTY")
@processor.post_run(validation.validation_keep_contradiction)
@click.pass_context
def transform_negate(ctx, datasets, batch_size, no_gpu):
    """Negates the received sentences with polyjuice.

    This operation is a transformation.
    It tries to negate the sentence if possible.
    This transformation is only available for to-english datasets.
    """
    total_records = sum(
        len(orig["records"]) for orig in datasets if orig["lang"] == "en"
    )
    if total_records == 0:
        click.echo(fmt.no_records_message("Negation"))
        return datasets

    transf_id = "transf-negation"

    ctx.obj.register_transform(transf_id)

    gpu = accelerator.use_gpu(no_gpu)

    neg_polyjuice = model.NegPolyjuice(cuda=gpu)
    transf = transform.Negation(
        neg_polyjuice=neg_polyjuice, original_field="original", critical_field=transf_id
    )

    pbar = fmt.pbar_from_total(total_records, "Negation")

    processed = []
    for orig in datasets:
        if orig["lang"] == "en":
            old_records = orig["records"]
            new_records = []

            for i in range(0, len(old_records), batch_size):
                batch = old_records[i : i + batch_size]
                records = transf(batch)
                new_records.extend(records)
                pbar.update(len(batch))

            orig["records"] = new_records

        processed.append(orig)

    return processed
