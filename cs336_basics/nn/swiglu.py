import torch
import einops
from cs336_basics.nn.linear import Linear


class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.d_ff = d_ff
        self.d_model = d_model

        self.w1 = Linear(
            in_features=d_model, out_features=self.d_ff, device=device, dtype=dtype
        )

        self.w2 = Linear(
            in_features=self.d_ff, out_features=d_model, dtype=dtype, device=device
        )

        self.w3 = Linear(
            in_features=d_model, out_features=self.d_ff, device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        silu = torch.nn.functional.sigmoid(self.w1(x)) * self.w1(x)
        swiglu = self.w2(silu * self.w3(x))

        return swiglu
