"""Extractors: turn audio files into feature vectors.

All extractors pool into 1-second chunks so that distance computation
operates on per-second feature vectors, not per-file.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class Extractor(ABC):
    """Abstract base for a feature extractor."""

    name: str
    version: str = "1"

    def __init__(self, device: str = "cuda", cache_dir: str | Path = "~/.cache/srm-eval/features"):
        self.device = device
        self.cache_dir = Path(cache_dir).expanduser() / self.name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def extract(self, path: Path) -> np.ndarray:
        """Return a (T, D) array — one D-dim vector per 1-second chunk."""

    def extract_cached(self, path: Path) -> np.ndarray:
        """Extract with on-disk caching."""
        h = hashlib.md5(f"{self.name}:{self.version}:{path}".encode()).hexdigest()
        cache_path = self.cache_dir / f"{h}.npy"
        if cache_path.exists():
            return np.load(cache_path)
        feats = self.extract(path)
        np.save(cache_path, feats)
        return feats

    def extract_system(
        self,
        file_df,
        *,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Extract features for all files in a system, returning (N_chunks, D)."""
        chunks: list[np.ndarray] = []
        paths = [Path(p) for p in file_df["filepath"]]
        iterator = paths
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(paths, desc=f"extracting {self.name}")
        for p in iterator:
            try:
                feats = self.extract_cached(p)
                if feats.ndim == 1:
                    feats = feats.reshape(1, -1)
                chunks.append(feats)
            except Exception as e:
                print(f"WARNING: {p}: {e}")
        if not chunks:
            return np.zeros((0, 1))
        return np.concatenate(chunks, axis=0)
