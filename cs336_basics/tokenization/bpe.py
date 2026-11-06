import os
from typing import List, Dict, Tuple, Iterator
from argparse import ArgumentParser
import pickle
from tqdm import tqdm
from cs336_basics.tokenization.pretokenization import (
    pre_tokenize_file,
    pre_tokenize_text,
)


class BPETrainer:

    def __init__(
        self,
        vocab_size: int,
        special_tokens: List[str] | None = None,
        num_processors=4,
        pbar=False,
    ):

        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.num_processors = num_processors
        self.pbar = pbar

    def train(self, input_path):
        self.vocab = {i: bytes((i,)) for i in range(256)}

        for token in set(self.special_tokens):
            self.vocab[len(self.vocab)] = token.encode("utf-8")

        self.merges = []
        self.corpus = []

        self.counts = pre_tokenize_file(
            input_path, self.special_tokens, num_processes=self.num_processors
        )

        self.occr = {}
        for pre_token in self.counts:
            self.corpus.append([self.counts[pre_token], pre_token, list(pre_token)])
            pre_token_id = len(self.corpus) - 1
            for i in range(len(pre_token) - 1):
                j = i + 1
                key = (pre_token[i], pre_token[j])
                if key in self.occr:
                    self.add_subword(key, pre_token_id)
                else:
                    self.occr[key] = (self.counts[pre_token], {pre_token_id: 1})

        with tqdm(
            total=self.vocab_size - len(self.vocab),
            desc="Training...",
            disable=not self.pbar,
        ) as progress:
            while self.vocab_size > len(self.vocab):
                merge = self.__get_best_merge()
                self._perform_merge(merge)
                progress.update(1)

        print(f"[Tokenizer] Done training tokenizer with vocab size {len(self.vocab)}")

    def __get_best_merge(self):
        return max(
            self.occr,
            key=lambda pair: (
                self.occr[pair][0],
                self.vocab[pair[0]],
                self.vocab[pair[1]],
            ),
        )

    def remove_subword(self, subword, word_id):
        pre_token = self.corpus[word_id][1]
        current_count, dict_occr = self.occr[subword]
        dict_occr[word_id] -= 1
        self.occr[subword] = (current_count - self.counts[pre_token], dict_occr)

    def add_subword(self, subword, word_id):
        pre_token = self.corpus[word_id][1]
        if subword in self.occr:
            current_count, dict_occr = self.occr[subword]
            dict_occr[word_id] = dict_occr.get(word_id, 0) + 1
            self.occr[subword] = (current_count + self.counts[pre_token], dict_occr)
        else:
            self.occr[subword] = (self.counts[pre_token], {word_id: 1})

    def _perform_merge(self, merge):
        new_token_id = len(self.vocab)
        for word_id in set(self.occr[merge][1]):
            count, pre_token, old_tokens = self.corpus[word_id]
            i = 0
            new_tokens = []
            subtractions = set()
            additions = set()
            while i < len(old_tokens) - 1:
                j = i + 1
                if (old_tokens[i], old_tokens[j]) == merge:
                    if i > 0:
                        prev_subword = (i - 1, i)
                        subtractions.add(prev_subword)
                    if i < len(old_tokens) - 2:
                        post_subword = (i + 1, i + 2)
                        subtractions.add(post_subword)
                    new_tokens.append(new_token_id)
                    i += 2
                else:
                    new_tokens.append(old_tokens[i])
                    i += 1

            if i < len(old_tokens):
                new_tokens.append(old_tokens[-1])

            for i, j in subtractions:
                subword = (old_tokens[i], old_tokens[j])
                self.remove_subword(subword, word_id)

            i = 0
            while i < len(new_tokens):
                if new_tokens[i] == new_token_id:
                    if i > 0:
                        subword = (i - 1, i)
                        additions.add(subword)
                    if i < len(new_tokens) - 1:
                        subword = (i, i + 1)
                        additions.add(subword)
                i += 1

            for i, j in additions:
                subword = (new_tokens[i], new_tokens[j])
                self.add_subword(subword, word_id)

            self.corpus[word_id] = (count, pre_token, new_tokens)

        self.vocab[new_token_id] = self.vocab[merge[0]] + self.vocab[merge[1]]
        self.occr.pop(merge)
        self.merges.append((self.vocab[merge[0]], self.vocab[merge[1]]))

    def save_tokenizer(self, folder):
        with open(os.path.join(folder, "vocab.pkl"), "wb") as f:
            pickle.dump(self.vocab, f)

        with open(os.path.join(folder, "merges.pkl"), "wb") as f:
            pickle.dump(self.merges, f)

        print("[Tokenizer] Saved tokenizer in: ", folder)


class Tokenizer:
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: List[str] | None = None,
    ):
        self.vocab = vocab
        self.tok2ind = {value: key for key, value in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens

    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: List[str] | None = None,
    ):
        raise NotImplementedError()

    def encode(self, text: str) -> List[int]:
        pre_tokens = pre_tokenize_text(text, special_tokens=self.special_tokens)
        tokens = []
        for word in pre_tokens:
            if self.special_tokens and word.decode("utf-8") in self.special_tokens:
                tokens.append(self.tok2ind[word])
                continue

            word = [self.tok2ind[bytes((t,))] for t in word]
            for t1, t2 in self.merges:
                tok = t1 + t2
                t1 = self.tok2ind[t1]
                t2 = self.tok2ind[t2]
                temp_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == (t1, t2):
                        temp_word.append(self.tok2ind[tok])
                        i += 2
                    else:
                        temp_word.append(word[i])
                        i += 1
                word = tuple(temp_word)
            tokens.extend(word)
        return tokens

    def encode_iterable(self, iterable: Iterator[str]) -> Iterator[int]:
        buffer = ""

        for chunk in iterable:
            buffer += chunk

            if self.special_tokens:
                for special_token in self.special_tokens:
                    idx = buffer.find(special_token)
                    if idx != -1:
                        end = idx + len(special_token)
                        yield from self.encode(buffer[:end])
                        buffer = buffer[end:]

        if buffer:
            yield from self.encode(buffer)

    def decode(self, ids: List[int]) -> str:
        output = b""

        for id in ids:
            output += self.vocab[id]

        return output.decode("utf-8", errors="replace")


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--input_path", required=True)
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--vocab_size", type=int, default=500)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--special_tokens", nargs="*", type=str, default=["<|endoftext|>"]
    )

    args = parser.parse_args()

    print(
        f"[Tokenizer] Training tokenizer with {args.vocab_size} vocabs, and special tokens {args.special_tokens}"
    )
    tokenizer = BPETrainer(
        args.vocab_size, args.special_tokens, num_processors=args.nproc, pbar=True
    )
    tokenizer.train(args.input_path)
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save_tokenizer(args.output_dir)
