import torch
import einops
from cs336_basics.nn.linear import Linear
from cs336_basics.nn.functional import scaled_dot_product_attention

from jaxtyping import Float
from torch import Tensor


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, pos_emb=None, device=None, dtype=None):
        super().__init__()

        self.device = device
        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model // num_heads
        self.dv = d_model // num_heads

        self.pos_emb = pos_emb
        self.lk = Linear(
            in_features=d_model,
            out_features=self.dk * num_heads,
            device=device,
            dtype=dtype,
        )

        self.lq = Linear(
            in_features=d_model,
            out_features=num_heads * self.dk,
            device=device,
            dtype=dtype,
        )

        self.lv = Linear(
            in_features=d_model,
            out_features=self.dv * num_heads,
            device=device,
            dtype=dtype,
        )

        self.lo = Linear(
            in_features=num_heads * self.dv,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

    def forward(self, x) -> Float[Tensor, " ... sequence_length d_model"]:
        """
            x: tensor of shape " ... sequence_length d_model"
        Returns:
            tensor of shape " ... sequence_length d_model"
        """
        seq_len = x.shape[-2]
        Q = self.lq(x)
        K = self.lk(x)
        V = self.lv(x)

        Q = einops.rearrange(
            Q, "... seq_len (n dq) -> ... n seq_len dq", n=self.num_heads
        )
        K = einops.rearrange(
            K, "... seq_len (n dk) -> ... n seq_len dk", n=self.num_heads
        )
        V = einops.rearrange(
            V, "... seq_len (n dv) -> ... n seq_len dv", n=self.num_heads
        )

        if self.pos_emb:
            token_positions = torch.arange(
                end=seq_len, device=self.device, dtype=torch.int32
            )
            token_positions = token_positions.expand(Q.shape[:-1])
            Q = self.pos_emb(Q, token_positions)
            K = self.pos_emb(K, token_positions)

        mask = torch.ones((seq_len, seq_len), device=self.device)
        mask = ~torch.triu(mask, diagonal=1).to(dtype=torch.bool)

        attn = scaled_dot_product_attention(K, Q, V, mask=mask)

        attn = einops.rearrange(
            attn, "... n seq_len dv -> ... seq_len (n dv)", n=self.num_heads
        )
        o_proj = self.lo(attn)
        return o_proj
