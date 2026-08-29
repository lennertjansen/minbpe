# minbpe

**Disclaimer: this repo is strictly for my own educational purposes and understanding of byte-pair encoding (BPE) for LLM tokenization.**

My from-scratch implementation of [Karpathy's minbpe](https://github.com/karpathy/minbpe): a byte-level BPE tokenizer as used for LLMs (GPT-2 style). `BasicTokenizer` trains a merge table on raw UTF-8 bytes; no regex splitting or special tokens yet.

Built following:
- ["Let's build the GPT Tokenizer"](https://www.youtube.com/watch?v=zduSFxRajkE), with `tokenization.ipynb` as the scratchpad along the way;
- [Wikipedia article about BPE](https://en.wikipedia.org/wiki/Byte-pair_encoding);
- GPT-2 [paper (section 2.2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) and [code](https://github.com/openai/gpt-2/blob/master/src/encoder.py);
- [Sennrich et al. (2016). "Neural Machine Translation of Rare Words with Subword Units". ACL, §3.2](https://arxiv.org/pdf/1508.07909)

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

## AI statement

No AI was used to write the code. Claude Code (Fable 5) drafted and edited this README. Cursor (GPT-5.6 (sol, medium)) reviewed and refined my handwritten notes and comments in the code and notebooks.
