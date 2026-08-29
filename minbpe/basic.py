"""
A basic implementation of a (byte-level) Byte Pair Encoding tokenizer for LLMs.

https://en.wikipedia.org/wiki/Byte-pair_encoding

Follows the GPT-2 tokenizer.
- Code:
    - https://github.com/openai/gpt-2/blob/master/src/encoder.py 
- Paper (Section 2.2):
    - https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

Diverges from the actual GPT-2 tokenizer in that it:
- Doesn't handle regex splitting patterns for enforcing merges that can never happen.
- Doesn't handle special tokens like <|endoftext|> or UNK.
"""

from .base import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()
        self.base_vocab = 256

    def train(self, text: str, vocab_size: int, verbose=False):
        assert vocab_size >= self.base_vocab
        num_merges = vocab_size - self.base_vocab

        # process input text
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes) # list of integers in range 0..255

        # iteratively merge most common adjacent pairs to create new tokens
        merges: dict[tuple[int, int], int] = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            
            # count number of times each consecutive pair appears
            stats = get_stats(ids)
            if not stats:
                raise ValueError(
                    f"cannot reach vocab_size={vocab_size}; "
                    f"no pairs left at {self.base_vocab + i} tokens"
                )
            
            # find pair with highest count
            pair = max(stats, key=stats.get)

            # assign the new token the next available id
            idx = self.base_vocab + i

            # replace all occurrences of pair in ids with the new token (idx)
            ids = merge(ids, pair, idx)

            # updates the merges and vocab tables
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(f"Merge {i + 1}/{num_merges}: {pair} -> into a new token {idx} | ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges # used in self.encode()
        self.vocab = vocab # used in self.decode()

    def encode(self, text: str):
        """Given a string (text), return a list of token ids."""

        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of ints in range 0..255
        
        while len(ids) >= 2:
            # find pair with lowest merge index, so we can apply earliest learned rule
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf"))) 
            if pair not in self.merges:
                    break # nothing else can be merged
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def decode(self, ids):
        """Given a list of ints (ids) return the decoded string (text)."""
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors='replace')
        return text