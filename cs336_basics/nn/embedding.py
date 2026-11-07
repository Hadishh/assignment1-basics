import torch
import einops


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        embedding = torch.zeros(
            (num_embeddings, embedding_dim), device=device, dtype=dtype
        )
        torch.nn.init.trunc_normal_(embedding, mean=0, std=1, a=-3, b=3)

        self.w = torch.nn.Parameter(
            data=embedding,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.w[token_ids]


class RotaryPositionalembedding(torch.nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.theta = torch.Tensor([theta], device=device)
        self.d_k = d_k

        I = torch.arange(end=max_seq_len, device=device)
        K = torch.arange(end=d_k // 2, device=device)
        denominator = self.theta.pow(-2 * K / d_k)
        theta_i_k = einops.einsum(
            I, denominator, "max_seq_len, half_d -> max_seq_len half_d"
        )

        R = torch.zeros((max_seq_len, d_k // 2, 2, 2))

        R[:, :, 0, 0] = torch.cos(theta_i_k)
        R[:, :, 0, 1] = -torch.sin(theta_i_k)
        R[:, :, 1, 0] = torch.sin(theta_i_k)
        R[:, :, 1, 1] = torch.cos(theta_i_k)

        self.register_buffer(name="R", tensor=R, persistent=False)

    def forward(self, x: torch.Tensor, token_positionts: torch.Tensor) -> torch.Tensor:
        R = self.R[token_positionts]
        x = einops.rearrange(x, "... seq_len (k p) -> ... seq_len k p", p=2)
        x = einops.einsum(R, x, "seq_len k l p, ... seq_len k p -> ... seq_len k l")
        x = einops.rearrange(x, "... seq_len k p -> ... seq_len (k p)")
        return x
