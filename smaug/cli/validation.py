import click
import functools
import re

from smaug import core
from smaug import functional
from smaug import models
from smaug import ops
from smaug import pipeline
from smaug.cli import accelerator
from smaug.cli import fmt
from smaug.cli import processor


_RM_EQ_CMD = "val-rm-eq"
_RM_PATTERN_CMD = "val-rm-pattern"
_KEEP_CONTRADICTION_CMD = "val-keep-contradiction"
_KEEP_EQ_NUM_CMD = "val-keep-eq-num"
_KEEP_EQ_NE_CMD = "val-keep-eq-ne"
_KEEP_GEQ_EDIT_DIST_CMD = "val-keep-geq-edit-dist"
_KEEP_LEQ_CHAR_INSERT_CMD = "val-keep-leq-char-ins"


@click.command(_RM_EQ_CMD, short_help="Remove sythetic records equal to the original.")
@click.option(
    "--transform",
    "cli_transforms",
    multiple=True,
    help="Transforms to filter with this validation. If not specified all are validated.",
)
@processor.make
@click.pass_context
def rm_eq(ctx, datasets, cli_transforms):
    """Validates if the generated records are not equal to the original.

    This operation is a validation. It ensures the generated record is different
    from the original one by performing a string comparison.
    """
    transforms = cli_transforms if cli_transforms else list(ctx.obj.iter_transforms())

    total_records = sum(len(orig["records"]) for orig in datasets)
    if total_records == 0:
        click.echo(fmt.no_records_message("Remove Equal"))
        return datasets

    processed = [dataset for dataset in datasets]

    for transform in transforms:
        pbar = fmt.pbar_from_total(total_records, f"Remove Equal for {transform}")
        validation_func = functional.lift_boolean_validation(lambda o, p: o != p)
        pipeline_func = pipeline.lift_validation(validation_func, transform)

        for dataset in processed:
            not_validated = core.Data(dataset["records"])
            dataset["records"] = pipeline_func(not_validated)
            pbar.update(len(not_validated))

    return processed


@click.command(
    _RM_PATTERN_CMD, short_help="Remove synthetic records that match a pattern."
)
@click.option("-p", "--pattern", required=True, help="Pattern to search.")
@click.option(
    "--transform",
    "cli_transforms",
    multiple=True,
    help="Transforms to filter with this validation. If not specified all are validated.",
)
@processor.make
@click.pass_context
def rm_pattern(ctx, datasets, pattern, cli_transforms):
    """Validates if the generated records do not match a regular expression.

    This operations is a validation. It ensures generated records do not have
    the given pattern.

    This validation is particularly useful with language models that may leave
    unwanted tokens after generation (such as masks or special tokens) to filter
    these occurrences.
    """

    transforms = cli_transforms if cli_transforms else list(ctx.obj.iter_transforms())

    total_records = sum(len(orig["records"]) for orig in datasets)
    if total_records == 0:
        click.echo(fmt.no_records_message(f'Remove Pattern "{pattern}"'))
        return datasets

    processed = [dataset for dataset in datasets]
    compiled_pattern = re.compile(pattern)
    for transform in transforms:
        pbar = fmt.pbar_from_total(
            total_records, f'Remove Pattern "{pattern}" for {transform}'
        )
        validation_func = functional.lift_boolean_validation(
            lambda o, p: compiled_pattern.search(p.value) is None,
        )
        pipeline_func = pipeline.lift_validation(validation_func, transform)
        for dataset in processed:
            not_validated = core.Data(dataset["records"])
            dataset["records"] = pipeline_func(not_validated)
            pbar.update(len(not_validated))

    return processed


@click.command(
    _KEEP_CONTRADICTION_CMD,
    help="Keep synthetic records contradicting the original.",
)
@click.option(
    "--transform",
    "cli_transforms",
    multiple=True,
    help="Transforms to filter with this validation. If not specified all are validated.",
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
def keep_contradiction(ctx, datasets, cli_transforms, batch_size, no_gpu):
    """Validates if the synthetic records contradict the original records.

    This operation is a validation. It uses a RoBERTa model trained for NLI
    to ensure the generated records contradict the original ones.
    """
    transforms = cli_transforms if cli_transforms else list(ctx.obj.iter_transforms())

    total_records = sum(len(orig["records"]) for orig in datasets)
    if total_records == 0:
        click.echo(fmt.no_records_message("Keep Contradictions"))
        return datasets

    gpu = accelerator.use_gpu(no_gpu)

    model, tokenizer = models.roberta_mnli_load()
    predict_func = functools.partial(
        ops.roberta_mnli_predict, model=model, tokenizer=tokenizer, cuda=gpu
    )
    contradiction_id = ops.roberta_mnli_contradiction_id(model)

    processed = [dataset for dataset in datasets]
    for transform in transforms:
        pbar = fmt.pbar_from_total(
            total_records, f"Keep Contradictions for {transform}"
        )
        validation_func = functional.lift_boolean_validation(
            lambda o, p: predict_func(f"{o} </s></s> {p}").argmax().item()
            == contradiction_id,
        )
        pipeline_func = pipeline.lift_validation(validation_func, transform)

        for dataset in processed:
            not_validated = dataset["records"]
            validated = []
            for i in range(0, len(not_validated), batch_size):
                batch = core.Data(not_validated[i : i + batch_size])
                validated.extend(pipeline_func(batch))
                pbar.update(len(batch))
            dataset["records"] = validated
    return processed


@click.command(
    _KEEP_EQ_NUM_CMD,
    help="Keep perturbations with the same numbers count as the original.",
)
@click.option(
    "--transform",
    "cli_transforms",
    multiple=True,
    help="Transforms to filter with this validation. If not specified all are validated.",
)
@processor.make
@click.pass_context
def keep_eq_num_count(ctx, datasets, cli_transforms):
    """Validates if the synthetic records have an equal number count as the original records.

    This operation is a validation. It uses Regular Expressions to detect the numbers
    both in the original and the perturbed sentences.
    """
    transforms = cli_transforms if cli_transforms else list(ctx.obj.iter_transforms())

    total_records = sum(len(orig["records"]) for orig in datasets)
    if total_records == 0:
        click.echo(fmt.no_records_message("Keep Equal Numbers Count"))
        return datasets

    processed = [dataset for dataset in datasets]
    for transform in transforms:
        pbar = fmt.pbar_from_total(
            total_records, f"Keep Equal Numbers Count for {transform}"
        )
        validation_func = functional.lift_boolean_validation(ops.equal_numbers_count)
        pipeline_func = pipeline.lift_validation(validation_func, transform)
        for dataset in processed:
            not_validated = core.Data(dataset["records"])
            dataset["records"] = pipeline_func(not_validated)
            pbar.update(len(not_validated))
    return processed


@click.command(
    _KEEP_EQ_NE_CMD,
    help="Keep perturbations with the same named entities count as the original.",
)
@click.option(
    "--transform",
    "cli_transforms",
    multiple=True,
    help="Transforms to filter with this validation. If not specified all are validated.",
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
def keep_eq_ne_count(ctx, datasets, cli_transforms, batch_size, no_gpu):
    """Validates if the synthetic records have an equal named entity count as the original records.

    This operation is a validation. It uses a Stanza NER model to detect the named entities
    both in the original and the perturbed sentences.
    """
    transforms = cli_transforms if cli_transforms else list(ctx.obj.iter_transforms())

    total_records = sum(
        len(dataset["records"])
        for dataset in datasets
        if models.stanza_ner_lang_available(dataset["lang"])
    )
    if total_records == 0:
        click.echo(fmt.no_records_message("Keep Equal Named Entities Count"))
        return datasets

    gpu = accelerator.use_gpu(no_gpu)

    processed = [dataset for dataset in datasets]
    for transform in transforms:
        pbar = fmt.pbar_from_total(
            total_records, f"Keep Equal Named Entities Count for {transform}"
        )
        for dataset in processed:
            lang = dataset["lang"]
            if not models.stanza_ner_lang_available(lang):
                continue
            ner_pipeline = models.stanza_ner_load(lang, gpu)
            validation_func = functools.partial(
                ops.equal_named_entities_count,
                ner_pipeline=ner_pipeline,
            )
            validation_func = functional.lift_boolean_validation(validation_func)
            pipeline_func = pipeline.lift_validation(validation_func, transform)
            not_validated = dataset["records"]
            validated = []
            for i in range(0, len(not_validated), batch_size):
                batch = core.Data(not_validated[i : i + batch_size])
                validated.extend(pipeline_func(batch))
                pbar.update(len(batch))
            dataset["records"] = validated
    return processed


@click.command(
    _KEEP_GEQ_EDIT_DIST_CMD,
    help="Keep perturbations with a minimum edit distance from the original above a threshold.",
)
@click.option(
    "-d", "--distance", type=int, required=True, help="Minimum threshold to accept."
)
@click.option(
    "-l",
    "--level",
    type=click.Choice(("char", "word"), case_sensitive=False),
    default="char",
    help="Level at which to measure the minimum edit distance.",
)
@click.option(
    "--transform",
    "cli_transforms",
    multiple=True,
    help="Transforms to filter with this validation. If not specified all are validated.",
)
@processor.make
@click.pass_context
def keep_geq_edit_dist(ctx, datasets, distance, level, cli_transforms):
    """Validates if the perturbations have a minimum edit distance higher than a threshold.

    This operation is a validation. It computes the minimum edit distance between the original
    and perturbed sentences.
    """
    transforms = cli_transforms if cli_transforms else list(ctx.obj.iter_transforms())

    total_records = sum(len(orig["records"]) for orig in datasets)
    if total_records == 0:
        click.echo(fmt.no_records_message(f"Keep Edit Distance above {distance}"))
        return datasets

    processed = [dataset for dataset in datasets]
    for transform in transforms:
        pbar = fmt.pbar_from_total(
            total_records, f"Keep Edit Distance above {distance} for {transform}"
        )

        validation_func = functional.lift_boolean_validation(
            lambda s1, s2: ops.edit_distance(s1, s2, level) >= distance
        )
        pipeline_func = pipeline.lift_validation(validation_func, transform)
        for dataset in processed:
            not_validated = core.Data(dataset["records"])
            dataset["records"] = pipeline_func(not_validated)
            pbar.update(len(not_validated))
    return processed


@click.command(
    _KEEP_LEQ_CHAR_INSERT_CMD,
    help="Keep perturbations with a total of char insertions below a threshold.",
)
@click.option(
    "-c",
    "--chars",
    default="<>()[]{}",
    show_default=True,
    help="Chars to consider (each individual char is considered)",
)
@click.option(
    "-i",
    "--max-insertions",
    type=int,
    required=True,
    help="Maximum insertions to accept.",
)
@click.option(
    "--transform",
    "cli_transforms",
    multiple=True,
    help="Transforms to filter with this validation. If not specified all are validated.",
)
@processor.make
@click.pass_context
def keep_leq_char_ins(ctx, datasets, chars, max_insertions, cli_transforms):
    """Validates if the pertubrations have a maximum number of character insertions.

    This operation is a validation. It computes the number of insertions of specific characters
    in the perturbed sentences, and only allows perturbations with this number bellow a threshold.
    """
    transforms = cli_transforms if cli_transforms else list(ctx.obj.iter_transforms())

    total_records = sum(len(orig["records"]) for orig in datasets)
    if total_records == 0:
        click.echo(
            fmt.no_records_message(f"Keep {chars} insertions below {max_insertions}")
        )
        return datasets

    processed = [dataset for dataset in datasets]
    for transform in transforms:
        pbar = fmt.pbar_from_total(
            total_records,
            f"Keep {chars} insertions below {max_insertions} for {transform}",
        )
        validation_func = functional.lift_boolean_validation(
            lambda o, p: ops.character_insertions(o, p, chars) <= max_insertions
        )
        pipeline_func = pipeline.lift_validation(validation_func, transform)
        for dataset in processed:
            not_validated = core.Data(dataset["records"])
            dataset["records"] = pipeline_func(not_validated)
            pbar.update(len(not_validated))
    return processed
