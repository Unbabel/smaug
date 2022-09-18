import click
import typing

from smaug import random
from smaug.cli import config
from smaug.cli import context
from smaug.cli import processor
from smaug.cli import io
from smaug.cli import transform
from smaug.cli import validation


@click.group(chain=True, invoke_without_command=True)
@click.option(
    "-c", "--cfg", type=str, help="Configuration file for the augmentation pipeline."
)
@click.option(
    "--no-post-run", is_flag=True, help="Disable default post runs for processors."
)
@click.option("-s", "--seed", type=int, help="Seed for reproducibility.")
def augment(cfg, no_post_run, seed):
    """Executes an augmentation pipeline with multiple operations.

    Transform operations generate synthetic records from original records.

    Validation operations verify if a synthetic record meets a desired criteria, removing it otherwise.

    Read and write operations for multiple formats are also available.
    """
    pass


@augment.result_callback()
@click.pass_context
def process_commands(
    ctx, processors, cfg: str, no_post_run: bool, seed: typing.Union[int, None]
):
    ctx.obj = context.Context()

    if cfg is not None:
        if len(processors) > 0:
            raise ValueError("No commands should be specified with --cfg argument.")
        _run_cfg_mode(cfg, no_post_run, seed)
    else:
        _run_chain_mode(ctx, processors, no_post_run, seed)


def _run_chain_mode(ctx, processors, no_post_run: bool, seed: typing.Union[int, None]):
    post_run = not no_post_run

    if seed:
        click.echo(f"Seed set to {seed}.")
        random.seed_everything(seed)

    # Start with an empty iterable.
    stream = ()

    # Pipe it through all stream processors.
    for proc in processors:
        stream = processor.call(ctx, proc, stream, post_run=post_run)

    # Evaluate the stream and throw away the items.
    _ = [r for r in stream]


def _run_cfg_mode(cfg: str, no_post_run: bool, seed: typing.Union[int, None]):
    args = config.to_args(cfg, no_post_run, seed)
    click.echo(f"Executing: augment {' '.join(args)}")
    augment(args)


augment.add_command(transform.delete_punct_span)
augment.add_command(transform.insert_text)
augment.add_command(transform.negate)
augment.add_command(transform.swap_ne)
augment.add_command(transform.swap_num)

augment.add_command(validation.keep_contradiction)
augment.add_command(validation.keep_eq_ne_count)
augment.add_command(validation.keep_eq_num_count)
augment.add_command(validation.keep_geq_edit_dist)
augment.add_command(validation.keep_leq_char_ins)
augment.add_command(validation.rm_eq)
augment.add_command(validation.rm_pattern)

augment.add_command(io.read_csv)
augment.add_command(io.read_lines)
augment.add_command(io.write_json)
