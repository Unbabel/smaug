import click

from maug import model
from maug import validation
from maug.cli import accelerator
from maug.cli import fmt
from maug.cli import processor


@click.command(
    "val-rm-equal", short_help="Remove sythetic records equal to the original."
)
@processor.make
@click.pass_context
def validation_remove_equal(ctx, datasets):
    """Validates if the generated records are not equal to the original.

    This operations is a validation. It ensures the generated record is different
    from the original one by performing a string comparison.
    """
    transforms = list(ctx.obj.iter_transforms())

    total_records = sum(len(orig["records"]) for orig in datasets)
    total_records *= len(transforms)
    if total_records == 0:
        click.echo(fmt.no_records_message("Remove Equal"))
        return datasets
    pbar = fmt.pbar_from_total(total_records, "Remove Equal")

    processed = [dataset for dataset in datasets]

    for transform in transforms:
        val = validation.NotEqual(original_field="original", critical_field=transform)

        for dataset in processed:
            not_validated = dataset["records"]
            dataset["records"] = val(not_validated)
            pbar.update(len(not_validated))

    return processed


@click.command(
    "val-rm-pattern", short_help="Remove synthetic records that match a pattern."
)
@click.option("-p", "--pattern", required=True, help="Pattern to search.")
@processor.make
@click.pass_context
def validation_remove_pattern(ctx, datasets, pattern):
    """Validates if the generated records do not match a regular expression.

    This operations is a validation. It ensures generated records do not have
    the given pattern.

    This validation is particularly usefull with language models that may leave
    unwanted tokens after generation (such as masks or special tokens) to filter
    these occurrences.
    """

    transforms = list(ctx.obj.iter_transforms())

    total_records = sum(len(orig["records"]) for orig in datasets)
    total_records *= len(transforms)
    if total_records == 0:
        click.echo(fmt.no_records_message(f'Remove Pattern "{pattern}"'))
        return datasets
    pbar = fmt.pbar_from_total(total_records, f'Remove Pattern "{pattern}"')

    processed = [dataset for dataset in datasets]
    for transform in transforms:
        val = validation.NoRegexMatch(
            pattern=pattern, original_field="original", critical_field=transform
        )
        for dataset in processed:
            not_validated = dataset["records"]
            dataset["records"] = val(not_validated)
            pbar.update(len(not_validated))

    return processed


@click.command(
    "val-keep-contradiction",
    help="Keep synthetic records contradicting the original.",
)
@click.option(
    "--batch-size",
    default=16,
    show_default=True,
    help="Batch size when processing records.",
)
@click.option("--no-gpu", is_flag=True, help="Disable gpu.")
@processor.make
@click.pass_context
def validation_keep_contradiction(ctx, datasets, batch_size, no_gpu):
    """Validates if the synthetic records contradict the original records.

    This operation is a validation. It uses a RoBERTa model trained for NLI
    to ensure the generated records contradict the original ones.
    """
    transforms = list(ctx.obj.iter_transforms())

    total_records = sum(len(orig["records"]) for orig in datasets)
    total_records *= len(transforms)
    if total_records == 0:
        click.echo(fmt.no_records_message("Keep Contradictions"))
        return datasets

    gpu = accelerator.use_gpu(no_gpu)

    roberta = model.RobertaMNLI(cuda=gpu)

    pbar = fmt.pbar_from_total(total_records, "Keep Contradictions")
    processed = [dataset for dataset in datasets]
    for transform in transforms:
        val = validation.IsContradiction(
            roberta, original_field="original", critical_field=transform
        )
        for dataset in processed:
            not_validated = dataset["records"]
            validated = []
            for i in range(0, len(not_validated), batch_size):
                batch = not_validated[i : i + batch_size]
                validated.extend(val(batch))
                pbar.update(len(batch))
            dataset["records"] = validated
    return processed