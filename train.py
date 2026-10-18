"""Train the GPT model from random initialization."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from config import GPTConfig
from model import GPTLanguageModel
from prepare_data import prepare_data

CHECKPOINT_DIR = Path("checkpoints")
DATA_PATH = Path("data/dataset.pt")
META_PATH = Path("data/meta.json")


def get_batch(split, data, batch_size, block_size, device):
    source = data[split]
    ix = torch.randint(len(source) - block_size - 1, (batch_size,))
    x = torch.stack([source[i : i + block_size] for i in ix])
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data, cfg, device):
    model.eval()
    result = {}
    for split in ("train", "val"):
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            xb, yb = get_batch(split, data, cfg.batch_size, cfg.block_size, device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        result[split] = losses.mean().item()
    model.train()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    args = parser.parse_args()

    cfg = GPTConfig()
    if args.max_iters is not None:
        cfg.max_iters = args.max_iters
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.eval_interval is not None:
        cfg.eval_interval = args.eval_interval

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if not DATA_PATH.exists() or not META_PATH.exists():
        prepare_data()

    data = torch.load(DATA_PATH, map_location="cpu", weights_only=True)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Vocabulary size: {meta['vocab_size']}")

    model = GPTLanguageModel(
        vocab_size=meta["vocab_size"],
        block_size=cfg.block_size,
        n_embd=cfg.n_embd,
        n_head=cfg.n_head,
        n_layer=cfg.n_layer,
        dropout=cfg.dropout,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()

    for step in range(cfg.max_iters + 1):
        if step % cfg.eval_interval == 0 or step == cfg.max_iters:
            losses = estimate_loss(model, data, cfg, device)
            elapsed = time.time() - start
            print(
                f"step {step:5d} | train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | elapsed {elapsed:.1f}s"
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg.__dict__,
                    "vocab_size": meta["vocab_size"],
                    "step": step,
                },
                CHECKPOINT_DIR / "gpt_checkpoint.pt",
            )

        if step == cfg.max_iters:
            break
        xb, yb = get_batch("train", data, cfg.batch_size, cfg.block_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

    print(f"Training finished. Checkpoint: {CHECKPOINT_DIR / 'gpt_checkpoint.pt'}")


if __name__ == "__main__":
    main()
