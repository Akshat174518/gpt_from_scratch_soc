# GPT From Scratch — Character-Level Transformer in PyTorch

A small decoder-only GPT language model implemented in PyTorch to learn how Transformer-based language models work internally. The project trains from random initialization on the Tiny Shakespeare text corpus and generates Shakespeare-style text.

> **Scope:** This is an educational, character-level language model. It is not ChatGPT, does not use pretrained weights, and is not intended for factual question answering.

## Features

- Character-level tokenizer built from the training corpus
- Decoder-only Transformer with causal (future-masked) self-attention
- Multi-head attention, feed-forward MLP, residual connections, LayerNorm, and dropout
- Train/validation split and cross-entropy loss
- AdamW optimization, gradient clipping, learning-rate configuration
- Checkpoint saving and text generation from a prompt
- Small automated tests for model shapes and causal masking

## Architecture

```text
Text corpus
    │
    ▼
Character tokenizer ──► integer token IDs
    │
    ▼
Token embeddings + positional embeddings
    │
    ▼
┌──────────────────────────────────┐
│ Transformer block × N             │
│  LayerNorm → causal self-attention│
│  residual connection              │
│  LayerNorm → feed-forward MLP     │
│  residual connection              │
└──────────────────────────────────┘
    │
    ▼
Final LayerNorm → vocabulary logits
    │
    ▼
Next-character prediction → autoregressive generation
```

At each position, causal attention allows the model to use the current and previous characters, but never future characters. Training uses the next character as the target.

## Repository structure

```text
gpt-from-scratch/
├── config.py              # Hyperparameters
├── model.py               # GPT model and Transformer blocks
├── prepare_data.py        # Download corpus, build vocabulary, save splits
├── train.py               # Training and validation loop
├── generate.py            # Load checkpoint and generate text
├── requirements.txt
├── .gitignore
├── LICENSE
├── README.md
├── data/
│   └── .gitkeep
└── tests/
    └── test_model.py
```

## Requirements

- Python 3.10+
- PyTorch
- NumPy
- Internet connection once, to download Tiny Shakespeare (or provide your own UTF-8 text file)

## Setup

```bash
git clone https://github.com/YOUR-USERNAME/gpt-from-scratch.git
cd gpt-from-scratch

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Train the model

The default configuration is intentionally small enough for CPU experimentation.

```bash
python prepare_data.py
python train.py
```

Training downloads Tiny Shakespeare into `data/input.txt` if it is not present, builds the character vocabulary, and writes checkpoints to `checkpoints/`.

To train for fewer steps while testing:

```bash
python train.py --max_iters 500
```

To override the batch size:

```bash
python train.py --batch_size 16
```

For GPU training, the script automatically uses CUDA when available. CPU training is supported but may be slow. You can change model size and training settings in `config.py`.

## Generate text

After training:

```bash
python generate.py --prompt "ROMEO:"
```

Change the length or sampling temperature:

```bash
python generate.py --prompt "To be, or not to be" --max_new_tokens 500 --temperature 0.8
```

A lower temperature makes sampling more conservative; a higher temperature makes it more varied. The model may generate malformed words, repeated phrases, or incoherent text—this is expected for a small model trained on a limited corpus.

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Key concepts

- **Tokenization:** Here each character is a token, making the vocabulary small and easy to inspect.
- **Next-token prediction:** Input `x = [c₁, c₂, ...]` is trained against shifted targets `y = [c₂, c₃, ...]`.
- **Causal self-attention:** A triangular mask prevents each position from attending to future positions.
- **Multi-head attention:** Several attention heads learn different context relationships in parallel.
- **Residual connections:** Add a sublayer's input to its output to support optimization.
- **Cross-entropy loss:** Measures how well the predicted distribution matches the next character.
- **Autoregressive generation:** Repeatedly predicts and samples one next character, appends it, and repeats.

## Results and honest reporting

Do not claim a particular loss, speed, or quality before running training on your machine. After training, record:
- device used (CPU/CUDA/MPS)
- training steps and time
- final training and validation loss
- a short generated sample
- limitations you observed

The checkpoint and downloaded dataset are excluded from Git by default. If you want reproducible results, record the configuration and random seed in your project notes.

## Possible extensions

1. Plot training and validation loss.
2. Add a command-line option for a custom text corpus.
3. Compare a bigram baseline against the Transformer.
4. Add top-k or top-p sampling.
5. Try a different public-domain corpus and compare validation loss.
6. Replace character tokenization with a subword tokenizer.
7. Experiment with model depth, number of heads, context length, and dropout.

## References

- Andrej Karpathy, [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- Karpathy's educational companion repository, [build-nanogpt](https://github.com/karpathy/build-nanogpt)
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

This repository is an educational implementation inspired by the concepts explained in the references above. Write your own notes, experiment with the code, and understand each component before presenting it as your work.
