import torch
import einops

from cs336_basics.nn.transformer import TransformerBlock
from cs336_basics.nn.embedding import RotaryPositionalembedding, Embedding
from cs336_basics.nn.norm import RMSNorm
from cs336_basics.nn.linear import Linear
from cs336_basics.nn.functional import softmax


class TranformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):

        super().__init__()
        self.n_layers = num_layers
        self.nheads = num_heads
        self.vocab_size = vocab_size

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype
        )

        self.rope = RotaryPositionalembedding(
            theta=rope_theta,
            d_k=d_model // num_heads,
            max_seq_len=context_length,
            device=device,
            dtype=dtype,
        )

        self.layers = torch.nn.ModuleList(
            [
                TransformerBlock(
                    num_heads=self.nheads,
                    d_model=d_model,
                    d_ff=d_ff,
                    pos_emb=self.rope,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.lm_head = Linear(
            in_features=d_model, out_features=vocab_size, device=device, dtype=dtype
        )

        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)

    def forward(self, tokens):

        x = self.token_embeddings(tokens)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        x = self.lm_head(x)

        return x
