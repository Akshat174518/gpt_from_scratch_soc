"""Generate text from a trained GPT checkpoint."""
import argparse
import json
from pathlib import Path

import torch

from model import GPTLanguageModel

CHECKPOINT_PATH = Path("checkpoints/gpt_checkpoint.pt")
META_PATH = Path("data/meta.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    if not CHECKPOINT_PATH.exists() or not META_PATH.exists():
        raise SystemExit("Checkpoint or vocabulary missing. Run `python prepare_data.py` and `python train.py` first.")

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    cfg = checkpoint["config"]
    model = GPTLanguageModel(
        vocab_size=checkpoint["vocab_size"],
        block_size=cfg["block_size"],
        n_embd=cfg["n_embd"],
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    stoi = meta["stoi"]
    itos = {int(i): ch for i, ch in meta["itos"].items()}
    unknown = sorted(set(args.prompt) - set(stoi))
    if unknown:
        raise SystemExit(f"Prompt contains characters outside the training vocabulary: {unknown!r}")
    prompt_ids = torch.tensor([[stoi[ch] for ch in args.prompt]], dtype=torch.long, device=device)
    output = model.generate(prompt_ids, args.max_new_tokens, args.temperature)[0].tolist()
    print("".join(itos[i] for i in output))


if __name__ == "__main__":
    main()
