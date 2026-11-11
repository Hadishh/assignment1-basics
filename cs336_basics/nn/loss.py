import torch
import einops
from cs336_basics.nn.functional import softmax


def cross_entropy_loss(o: torch.Tensor, y: torch.Tensor):
    """
    o: tensor of size (B, vocab)
    y: tensor of size (B,)
    """
    o = o - o.amax(dim=-1, keepdim=True)

    sum_exp = torch.exp(o).sum(dim=-1)
    selected_ = o.gather(dim=1, index=y.unsqueeze(1))
    logprops = selected_ - torch.log(sum_exp)

    return -logprops.mean()
