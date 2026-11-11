import numpy as np
import torch


def data_loading(sequence, batch_size, context_length, device="cpu"):
    start_inds = np.arange(0, len(sequence) - context_length)
    start_inds = np.random.choice(start_inds, size=batch_size, replace=False)

    indices = start_inds[:, None] + np.arange(context_length)[None, :]
    batch = sequence[indices]  # shape (B, L)

    output_toks = sequence[indices + 1]

    return torch.from_numpy(batch).to(
        device=device, dtype=torch.int64
    ), torch.from_numpy(output_toks).to(device=device, dtype=torch.int64)
