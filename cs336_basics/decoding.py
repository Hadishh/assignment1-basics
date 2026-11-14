import torch
from cs336_basics.nn.functional import softmax


def greedy_decoding(probs: torch.Tensor):
    return torch.argmax(probs, dim=-1, keepdim=True)


def top_p_sampling(probs: torch.Tensor, top_p: float):
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)

    cumsum = torch.cumsum(sorted_probs, dim=-1, dtype=torch.float32)
    keep = torch.ones_like(cumsum, dtype=torch.bool)
    keep[:, 1:] = cumsum[:, :-1] < top_p

    sorted_probs[~keep] = 0

    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    positions = torch.multinomial(sorted_probs, num_samples=1)
    sampled_tokens = indices.gather(dim=1, index=positions)

    return sampled_tokens


def convert_tensor_to_tokens(
    batch_sequence: torch.Tensor,
    stop_token: int,
    prompt_length: int,
):
    rows = batch_sequence.detach().cpu().tolist()
    results = []

    for row in rows:
        generated = row[prompt_length:]
        if stop_token in generated:
            end = prompt_length + generated.index(stop_token) + 1
            row = row[:end]
        results.append(row)

    return results


def decode(
    model,
    input_sequence: torch.Tensor,
    max_new_tokens: int,
    stop_token: int,
    temperature: float = 1,
    top_p: float = -1,
):
    """
    Args:
        model: transfomer model
        input_sequence: sequence of size (B, L)
        max_get: Maximum new tokens to be generated
        stop_token: stop token
        temperature: softmax temperature
        top_p: top p sampling parameter
    """
    B = input_sequence.shape[0]
    assert top_p <= 1, "Top-p should be lower than or equal to 1"
    seq_done = torch.zeros(size=(B, 1), dtype=torch.bool, device=input_sequence.device)

    sequence = input_sequence.clone()

    gen_idx = 0
    while (not seq_done.all()) and gen_idx < max_new_tokens:
        logits = model(sequence)
        logits = logits[:, -1, :]  # B, vocab_size
        probs = softmax(logits.float(), dim=-1, tau=temperature)
        if top_p > 0:
            new_tokens = top_p_sampling(probs, top_p=top_p)
        else:
            new_tokens = greedy_decoding(probs)  # B, 1

        sequence = torch.cat((sequence, new_tokens), dim=-1)
        gen_idx += 1
        seq_done = seq_done | (new_tokens == stop_token)

    return convert_tensor_to_tokens(
        sequence, prompt_length=input_sequence.shape[1], stop_token=stop_token
    )
