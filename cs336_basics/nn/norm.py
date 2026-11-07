import torch
import einops


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.eps = eps
        self.d_model = d_model
        self.weight = torch.nn.Parameter(
            torch.ones((d_model), device=device, dtype=dtype)
        )

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        in_type = a.dtype

        a = a.to(torch.float32)

        sq = a.pow(2)
        sum_ = einops.reduce(
            sq,
            "... d_model -> ... 1",
            reduction="sum",
        )

        rms = sum_ / self.d_model + self.eps
        rms = torch.sqrt(rms)

        rmsnorm = einops.einsum(
            a,
            self.weight,
            "... d_model, d_model -> ... d_model",
        )
        rmsnorm = rmsnorm / rms

        return rmsnorm.to(dtype=in_type)
