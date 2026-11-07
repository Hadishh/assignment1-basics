import torch
import einops
import math


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        std = 2 / (in_features + out_features)
        bound = math.sqrt(std) * 3

        w = torch.zeros(size=(out_features, in_features), device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(w, mean=0, std=std, a=-bound, b=bound)

        self.w = torch.nn.Parameter(data=w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return einops.einsum(x, self.w, "... d_in, d_out d_in -> ... d_out")
