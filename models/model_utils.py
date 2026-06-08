import hashlib
import os
import random
from typing import Sequence

import numpy as np


def seed_everything(seed: int = 42, deterministic_torch: bool = True) -> int:
    """Seed random / numpy / torch for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed


def make_rng(seed: int = 42) -> np.random.Generator:
    """Return a fresh, independent numpy Generator (preferred over global state)."""
    return np.random.default_rng(seed)


def get_model_device(model):
    """Return the device of a torch module's first parameter."""
    return next(model.parameters()).device


def count_parameters(model) -> int:
    """Number of trainable parameters in a torch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


class OfflineTextEncoder:
    """Deterministic hashing bag-of-words text encoder (no torch, no downloads).

    Uses a fixed BLAKE2b digest (not Python's salted ``hash``) so embeddings are
    identical across processes -- a drop-in offline stand-in for Sentence-BERT.
    """

    def __init__(self, dim: int = 384, seed: int = 42):
        self.dim = dim
        self.seed = seed

    def _token_index(self, token: str) -> int:
        digest = hashlib.blake2b(f"{self.seed}:{token}".encode("utf-8"),
                                 digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for i, text in enumerate(texts):
            for tok in text.lower().split():
                if tok.isalpha():
                    out[i, self._token_index(tok)] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(norms, 1e-12, None)
