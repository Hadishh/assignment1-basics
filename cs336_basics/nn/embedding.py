import torch


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
