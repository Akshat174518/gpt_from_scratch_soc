"""Default hyperparameters for the small character-level GPT."""
from dataclasses import dataclass


@dataclass
class GPTConfig:
    batch_size: int = 32
    block_size: int = 128
    max_iters: int = 3000
    eval_interval: int = 300
    eval_iters: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1
    seed: int = 1337
