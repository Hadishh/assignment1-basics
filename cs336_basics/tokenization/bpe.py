import os
from typing import List
from argparse import ArgumentParser
import pickle
from tqdm import tqdm
from cs336_basics.tokenization.pretokenization import pre_tokenize

class Tokenizer:
                
    def __init__(self, vocab_size: int, special_tokens: List[str], num_processors=4, pbar=False):
        
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.num_processors = num_processors
        self.pbar = pbar


    def from_pretrained(self, folder_path):
        raise NotImplementedError

    
    def train(self, input_path):
        self.vocab = {i: bytes((i,)) for i in range(256)}
        
        for token in set(self.special_tokens): 
            self.vocab[len(self.vocab)] = token.encode("utf-8")
        
        self.merges = []
        self.corpus = []
        
        self.counts = pre_tokenize(input_path, self.special_tokens, num_processes=self.num_processors)

        self.occr = {}
        for pre_token in self.counts:
            self.corpus.append([self.counts[pre_token], pre_token, list(pre_token)])
            pre_token_id = len(self.corpus) - 1
            for i in range(len(pre_token) - 1):
                j = i + 1
                key = (pre_token[i], pre_token[j])
                if key in self.occr:
                    current_count, list_occr = self.occr[key]
                    list_occr.append(pre_token_id)
                    self.occr[key] = (current_count + self.counts[pre_token], list_occr)
                else:
                    self.occr[key] = (self.counts[pre_token], [pre_token_id])

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
        pairs = [(key, value[0]) for key, value in self.occr.items()]
        pairs = sorted(pairs, key=lambda pair: pair[1], reverse=True)
        
        ties = [(pairs[0][0], (self.vocab[pairs[0][0][0]], self.vocab[pairs[0][0][1]]))]
        for i in range(1, len(pairs)):
            if pairs[i][1] == pairs[0][1]:
                ties.append(
                    (
                        pairs[i][0], 
                        (self.vocab[pairs[i][0][0]], self.vocab[pairs[i][0][1]])
                    )
                )
            else:
                break

        return sorted(ties, key=lambda x: x[1], reverse=True)[0][0]

    def remove_subword(self, subword, word_id):
        pre_token = self.corpus[word_id][1]
        current_count, list_occr = self.occr[subword]
        list_occr.remove(word_id)
        self.occr[subword] = (current_count - self.counts[pre_token], list_occr)


    def add_subword(self, subword, word_id):
        pre_token = self.corpus[word_id][1]
        if subword in self.occr:
            current_count, list_occr = self.occr[subword]
            list_occr.append(word_id)
            self.occr[subword] = (current_count + self.counts[pre_token], list_occr)
        else:
            self.occr[subword] = (self.counts[pre_token], [word_id])


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

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--input_path", required=True)
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--vocab_size", type=int, default=500)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--special_tokens", nargs="*", type=str, default=["<|endoftext|>"])


    args = parser.parse_args()

    print(f"[Tokenizer] Training tokenizer with {args.vocab_size} vocabs, and special tokens {args.special_tokens}")
    tokenizer = Tokenizer(args.vocab_size, args.special_tokens, num_processors=args.nproc, pbar=True)
    tokenizer.train(args.input_path)
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save_tokenizer(args.output_dir)

    