"""Download Tiny Shakespeare and prepare train/validation token IDs."""
import json
import urllib.request
from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path("data")
TEXT_PATH = DATA_DIR / "input.txt"
META_PATH = DATA_DIR / "meta.json"
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def prepare_data(force_download: bool = False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if force_download or not TEXT_PATH.exists():
        print("Downloading Tiny Shakespeare...")
        urllib.request.urlretrieve(DATA_URL, TEXT_PATH)

    text = TEXT_PATH.read_text(encoding="utf-8")
    if len(text) < 1000:
        raise ValueError("The corpus looks too small. Check data/input.txt.")

    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {str(i): ch for ch, i in stoi.items()}
    encoded = np.array([stoi[ch] for ch in text], dtype=np.int64)
    split = int(0.9 * len(encoded))
    train_data = torch.tensor(encoded[:split], dtype=torch.long)
    val_data = torch.tensor(encoded[split:], dtype=torch.long)
    torch.save({"train": train_data, "val": val_data}, DATA_DIR / "dataset.pt")
    META_PATH.write_text(
        json.dumps({"stoi": stoi, "itos": itos, "vocab_size": len(chars)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Corpus characters: {len(text):,}")
    print(f"Vocabulary size: {len(chars)}")
    print(f"Training characters: {len(train_data):,}; validation characters: {len(val_data):,}")
    print(f"Saved dataset to {DATA_DIR / 'dataset.pt'} and vocabulary to {META_PATH}")


if __name__ == "__main__":
    prepare_data()
