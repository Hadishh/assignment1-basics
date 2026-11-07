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

        self.l1 = Linear(
            in_features=d_model, out_features=self.d_ff, device=device, dtype=dtype
        )

        self.l2 = Linear(
            in_features=self.d_ff, out_features=d_model, dtype=dtype, device=device
        )

        self.l3 = Linear(
            in_features=d_model, out_features=self.d_ff, device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        silu = torch.nn.functional.sigmoid(self.l1(x)) * self.l1(x)
        swiglu = self.l2(silu * self.l3(x))

        return swiglu
