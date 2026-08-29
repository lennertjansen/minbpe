# minbpe

**Disclaimer: this repo is strictly for my own educational purposes - understanding the byte-pair encoding (BPE) algorithm.**

My from-scratch implementation of [Karpathy's minbpe](https://github.com/karpathy/minbpe): a byte-level BPE tokenizer as used for LLMs (GPT-2 style). `BasicTokenizer` trains a merge table on raw UTF-8 bytes; no regex splitting or special tokens.

Built following ["Let's build the GPT Tokenizer"](https://www.youtube.com/watch?v=zduSFxRajkE), with `tokenization.ipynb` as the scratchpad along the way.

## Installation

Everything runs through [uv](https://docs.astral.sh/uv/) (Python 3.11+):

```bash
uv sync
```

## Usage

```python
from minbpe import BasicTokenizer

tokenizer = BasicTokenizer()
tokenizer.train("aaabdaaabac", vocab_size=256 + 3)  # 256 byte tokens + 3 merges
print(tokenizer.encode("aaabdaaabac"))  # [258, 100, 258, 97, 99]
print(tokenizer.decode([258, 100, 258, 97, 99]))  # "aaabdaaabac"
tokenizer.save("toy")  # writes toy.model (for load()) and toy.vocab (human-readable)
```

## Running tests

Tests cover encode/decode round-trips (trained on the [Manhattan Project Wikipedia article](https://en.wikipedia.org/wiki/Manhattan_Project)) and the [Wikipedia BPE example](https://en.wikipedia.org/wiki/Byte_pair_encoding):

```bash
uv run pytest
```

## License

MIT

## AI statement

No AI was used to write the code. Claude Code (Fable 5) drafted and edited this README. Cursor (GPT-5.6 (sol, medium)) reviewed and refined my handwritten notes and comments in the code and notebooks.
