import torch
import einops

from cs336_basics.nn.multihead_attention import MultiHeadAttention
from cs336_basics.nn.swiglu import SwiGLU
from cs336_basics.nn.norm import RMSNorm


class TransformerBlock(torch.nn.Module):
    def __init__(self, num_heads, d_model, d_ff, pos_emb=None, device=None, dtype=None):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            pos_emb=pos_emb,
            device=device,
            dtype=dtype,
        )
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, dtype=dtype, device=device)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))

        x = x + self.ffn(self.ln2(x))

        return x
