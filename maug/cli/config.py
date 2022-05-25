from distutils.command.config import config
import click
import yaml

from typing import Any, Dict, List, Union


def to_args(cfg: str, no_post_run: bool, seed: Union[int, None]) -> List[str]:
    """Parses a config into a list of arguments to execute the augment command."""
    with open(cfg, "r") as fp:
        cfg = yaml.safe_load(fp)

    augment_cfg = _build_augment_cfg(cfg)
    if seed is not None:
        augment_cfg.seed = seed
    if no_post_run:
        augment_cfg.no_post_run = no_post_run
    return augment_cfg.gen_args()


def _build_augment_cfg(cfg: Dict[str, Any]) -> "_AugmentCfg":
    if "pipeline" not in cfg:
        click.echo("Please specify the desired pipeline commands.")
        return
    pipeline = cfg.pop("pipeline")
    cmd_cfgs = []
    for cmd in pipeline:
        _extend_cmd_cfgs(cmd_cfgs, cmd)
    pipeline_cfg = _PipelineCfg(*cmd_cfgs)

    seed = cfg.get("seed", None)

    no_post_run = cfg.get("no-post-run", False)

    return _AugmentCfg(pipeline_cfg, seed, no_post_run)


def _extend_cmd_cfgs(cmd_cfgs: "List[_CommandCfg]", cmd: Dict[str, Any]):
    name = cmd.pop("cmd", None)
    if name is None:
        click.echo("Plase specify cmd inside pipeline.")
        exit(1)

    validations = cmd.pop("validations", [])

    cmd_args = {k: v for k, v in cmd.items()}
    cmd_cfgs.append(_CommandCfg(name, **cmd_args))

    for val in validations:
        val_name = val.pop("cmd", None)
        if val_name is None:
            click.echo(f"Plase specify cmd inside {name} validations.")
            exit(1)

        val_args = {k: v for k, v in val.items()}
        val_args["transform"] = name

        cmd_cfgs.append(_CommandCfg(val_name, **val_args))


class _AugmentCfg:
    def __init__(
        self, pipeline_cfg: "_PipelineCfg", seed: Union[int, None], no_post_run: bool
    ):
        self._pipeline_cfg = pipeline_cfg
        self.seed = seed
        self.no_post_run = no_post_run

    def gen_args(self) -> List[str]:
        args = []
        if self.seed is not None:
            args.extend(("--seed", str(self.seed)))
        if self.no_post_run:
            args.append("--no-post-run")
        args.extend(self._pipeline_cfg.gen_args())
        return args


class _PipelineCfg:
    def __init__(self, *cmd_cfgs: "_CommandCfg"):
        self._cmd_cfgs = list(cmd_cfgs)

    def gen_args(self):
        for cfg in self._cmd_cfgs:
            yield from cfg.gen_args()


class _CommandCfg:
    def __init__(self, name, **kwargs):
        self._name = name
        self._kwargs = kwargs

    def gen_args(self):
        yield self._name
        for k, v in self._kwargs.items():
            yield f"--{str(k)}"
            yield str(v)
