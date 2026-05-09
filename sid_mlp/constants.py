"""Constants, key functions, and seed utility for SID-MLP."""

import random

CODEBOOK_SIZE = 256
DIGIT_OFFSETS = [1, 257, 513, 769]


def triplet_key_int(d1cb, d2cb, d3cb):
    return (d1cb * CODEBOOK_SIZE + d2cb) * CODEBOOK_SIZE + d3cb


def quad_key_int(d1cb, d2cb, d3cb, d4cb):
    return ((d1cb * CODEBOOK_SIZE + d2cb) * CODEBOOK_SIZE + d3cb) * CODEBOOK_SIZE + d4cb


def token_to_codebook(token_id: int, digit_pos: int) -> int:
    idx = token_id - DIGIT_OFFSETS[digit_pos]
    assert 0 <= idx < CODEBOOK_SIZE, f"token_to_codebook out of range: {token_id} pos={digit_pos} -> {idx}"
    return idx


def set_seed(seed: int):
    """Fix random seeds for reproducibility."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
