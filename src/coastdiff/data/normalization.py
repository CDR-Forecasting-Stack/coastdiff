"""Normalization stats (mean/std per channel) for model I/O."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .channels import IN_CH, INVARIANT_CH, OUT_CH


def _load_stats(stats: dict, variables: list[str], group: str):
    mean = np.array([stats[group][v]["mean"] for v in variables])[:, None, None].astype(np.float32)
    std = np.array([stats[group][v]["std"] for v in variables])[:, None, None].astype(np.float32)
    return mean, std


class Normalizer:
    """Reproduces the training-time normalization exactly:
    input = concat(coarse channels, invariant channels), normalized by
    per-channel mean/std; output likewise."""

    def __init__(self, stats_path: str | Path):
        with open(stats_path) as f:
            stats = json.load(f)
        input_mean, input_std = _load_stats(stats, IN_CH, "input")
        inv_mean, inv_std = _load_stats(stats, INVARIANT_CH, "invariant")
        self.input_mean = np.concatenate([input_mean, inv_mean], axis=0)  # (36,1,1)
        self.input_std = np.concatenate([input_std, inv_std], axis=0)
        self.output_mean, self.output_std = _load_stats(stats, OUT_CH, "output")  # (8,1,1)

    def normalize_input(self, x: np.ndarray) -> np.ndarray:
        return (x - self.input_mean) / self.input_std

    def denormalize_output(self, x: np.ndarray) -> np.ndarray:
        return x * self.output_std + self.output_mean
