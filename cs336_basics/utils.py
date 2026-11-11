from typing import Iterable
import torch
import math
import os
import typing


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
