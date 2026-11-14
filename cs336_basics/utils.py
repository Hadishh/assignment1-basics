from typing import Iterable
import torch
import math
import os
import typing
import yaml


def load_yaml(argparser):
    pre_args, _ = argparser.parse_known_args()

    if pre_args.config:
        with open(pre_args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if config is None:
            config = {}

        if not isinstance(config, dict):
            argparser.error("Config must contain a YAML mapping.")

        # Catch misspelled or unsupported config keys.
        unknown = config.keys() - vars(pre_args).keys()
        if unknown:
            argparser.error(f"Unknown config keys: {list(unknown)}")

        argparser.set_defaults(**config)

    # Second pass: explicit CLI arguments override YAML defaults.
    args = argparser.parse_args()

    for name in ("training_dset", "output_dir"):
        if getattr(args, name) is None:
            argparser.error(f"Provide {name} in YAML or through --{name}.")

    return args


def lr_cosine_schedule(t, alpha_max, alpha_min, t_w, t_c):

    if t < t_w:
        return (t * alpha_max) / t_w

    elif t <= t_c:
        cosine = math.cos(((t - t_w) * math.pi) / (t_c - t_w))
        cosine = 0.5 * (1 + cosine)

        return alpha_min + cosine * (alpha_max - alpha_min)

    else:
        # t > t_c
        return alpha_min


def gradient_clipping(params: Iterable[torch.nn.Parameter], M: float, eps=1e-6):
    grads = [p.grad for p in params if p.grad is not None]

    l2_norm = torch.sqrt(sum([g.pow(2).sum() for g in grads]))

    if M < l2_norm:
        for g in grads:
            g.mul_(M / (eps + l2_norm))

    return l2_norm


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):

    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }

    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):

    obj = torch.load(src)

    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])

    return obj["iteration"]
