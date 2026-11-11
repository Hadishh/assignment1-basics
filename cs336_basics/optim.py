from collections.abc import Callable, Iterable
from cs336_basics.lr_scheduler import LRScheduler
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        betas,
        weight_decay,
        eps=1e-8,
        scheulder: LRScheduler | None = None,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        self.scheduler = scheulder
        defaults = {
            "lr": lr,
            "betas": betas,
            "epsilon": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            epsilon = group["epsilon"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                if self.scheduler:
                    lr = self.scheduler.calculate(t)

                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))

                alpha_t = lr * (math.sqrt(1 - beta2**t) / (1 - beta1**t))

                p.data -= lr * weight_decay * p.data

                m = beta1 * m + (1 - beta1) * p.grad.data
                v = beta2 * v + (1 - beta2) * p.grad.data.pow(2)

                p.data -= alpha_t * m / (torch.sqrt(v) + epsilon)

                state["m"] = m
                state["v"] = v
                state["t"] = t + 1

        return loss
