import os
import regex as re
from typing import BinaryIO, List
import multiprocessing as mp

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: List[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_tokens, list), "Must represent special token as a list of bytestring"
    if len(split_special_tokens) > 0:
        for item in split_special_tokens:
            assert isinstance(item, bytes)

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = -1
            for special_token in split_special_tokens:
                found_at = max(found_at, mini_chunk.find(special_token))
            
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _worker(path, start, end, special_tokens):
    counts = {}
    with open(path, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8")
        parts = re.split("|".join(map(re.escape, special_tokens)), chunk)

        for part in parts:
            for match_ in re.finditer(PAT, part):
                pre_token = match_.group()

                counts[pre_token] = counts.get(pre_token, 0) + 1
    return counts

def pre_tokenize(path: str | os.PathLike, special_tokens: List[str], num_processes=1):
    counts = {}
    special_tokens_str = [item for item in special_tokens]
    special_tokens = [item.encode("utf-8") for item in special_tokens]
    with open(path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens)

    args = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        args.append(
            (path, start, end, special_tokens_str)
        )

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(args)) as pool:
        results = pool.starmap(_worker, args)

    for out in results:
        for pre_token in out:
            enc = pre_token.encode("utf-8")
            counts[enc] = counts.get(enc, 0) + out[pre_token]

    return counts

    
                
    
