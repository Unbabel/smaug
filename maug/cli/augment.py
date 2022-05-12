import click
import typing

from maug import random
from maug.cli import context
from maug.cli import processor
from maug.cli import read
from maug.cli import transform
from maug.cli import validation
from maug.cli import write


@click.group(chain=True)
@click.option(
    "--no-post-run", is_flag=True, help="Disable default post runs for processors."
)
@click.option("-s", "--seed", type=int, help="Seed for reproducibility.")
def augment(no_post_run, seed):
    """Executes an augmentation pipeline with multiple operations.

    Transform operations generate synthetic records from original records.

    Validation operations verify if a synthetic record meets a desired criteria, removing it otherwise.

    Read and write operations for multiple formats are also available.
    """
    pass


@augment.result_callback()
@click.pass_context
def process_commands(ctx, processors, no_post_run: bool, seed: typing.Union[int, None]):
    ctx.obj = context.Context()

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


augment.add_command(read.read_csv)
augment.add_command(read.read_lines)

augment.add_command(transform.delete_punct_span)
augment.add_command(transform.negate)
augment.add_command(transform.swap_ne)
augment.add_command(transform.swap_num)

augment.add_command(validation.keep_contradiction)
augment.add_command(validation.keep_geq_edit_dist)
augment.add_command(validation.keep_eq_ne_count)
augment.add_command(validation.keep_eq_num_count)
augment.add_command(validation.rm_eq)
augment.add_command(validation.rm_pattern)

augment.add_command(write.write_json)
