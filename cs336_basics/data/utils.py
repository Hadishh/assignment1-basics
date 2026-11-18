import numpy as np
import torch

np.random.default_rng()


def data_loading(
    sequence,
    batch_size,
    context_length,
    device="cpu",
    rng: np.random.Generator | None = None,
):
    if rng is None:
        rng = np.random.default_rng()

    n_starts = len(sequence) - context_length

    start_inds = rng.choice(
        n_starts,
        size=batch_size,
        replace=False,
    )

    indices = start_inds[:, None] + np.arange(context_length)[None, :]
    batch = sequence[indices]
    output_toks = sequence[indices + 1]

    return (
        torch.from_numpy(batch).to(device=device, dtype=torch.int64),
        torch.from_numpy(output_toks).to(device=device, dtype=torch.int64),
    )
