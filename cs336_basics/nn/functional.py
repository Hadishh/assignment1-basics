import torch
import einops
import math


def softmax(v: torch.Tensor, dim: int) -> torch.Tensor:
    max_ = torch.amax(v, dim=dim, keepdim=True)

    v = v - max_
    sum_ = torch.sum(torch.exp(v), dim=dim, keepdim=True)

    v = torch.exp(v) / sum_

    return v


def scaled_dot_product_attention(
    K: torch.Tensor, Q: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None
):
    d = K.shape[-1]

    dot_prod = einops.einsum(Q, K, "... q d, ... k d -> ... q k")
    if mask is not None:
        dot_prod = dot_prod.masked_fill(~mask, float("-inf"))

    dot_prod /= math.sqrt(d)
    dot_prod = softmax(dot_prod, dim=-1)

    return einops.einsum(dot_prod, V, "... q k, ... k dv -> ... q dv")
