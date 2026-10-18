"""On-the-fly sliding windows over per-file symbol tensors (E010+).

The historical modules materialize every window (``fixed_step_curve.collect``).
Here each file is stored once as a ``(L, 2)`` float32 tensor and windows are
gathered by index, so a 210-file pool with window 127 costs ~30 MB instead of
several GB. The global window index ``i`` maps to file ``i // (L - W + 1)`` and
offset ``i % (L - W + 1)``, which is exactly the row order of ``collect``; a
batch drawn with the same permutation therefore contains the same windows.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .core import load_channel_data
from .experiment import ber_from_iq
from .paths import DATA_DIR, E003_DIR, E004_DIR, MANIFEST_DIR, project_path

CACHE_DIR = DATA_DIR / "cache" / "symbols_f64"
SPLIT_CONFIG = E004_DIR / "config.json"
SEQUENCE_LENGTH = 16_384


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_files() -> dict[str, list[str]]:
    """Train pool / validation / test file names used by E004-E009."""
    config = json.loads(SPLIT_CONFIG.read_text(encoding="utf-8"))
    return {
        "train_pool": list(config["train_pool_files"]),
        "validation": list(config["validation_files"]),
        "test": list(config["test_files"]),
    }


def resolve_paths(names: list[str]) -> list[Path]:
    from .large_scale_pilot import get_unique_files

    pilot = json.loads((E003_DIR / "config.json").read_text(encoding="utf-8"))
    candidates = get_unique_files(project_path(pilot["dataset_root"]), MANIFEST_DIR / "data_audit_all.csv")
    by_name = {path.name: path for path in candidates.path}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise FileNotFoundError(f"{len(missing)} split files not found in data/raw/archive, e.g. {missing[:3]}")
    return [by_name[name] for name in names]


def _cache_path(csv_path: Path) -> Path:
    return CACHE_DIR / f"{csv_path.stem}.npy"


def load_symbols(csv_path: Path) -> np.ndarray:
    """Return ``(L, 4)`` float64 [tx_I, tx_Q, rx_I, rx_Q]; cached as .npy (read via mmap)."""
    cache = _cache_path(csv_path)
    if cache.is_file() and cache.stat().st_mtime >= csv_path.stat().st_mtime:
        return np.load(cache, mmap_mode="r")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = CACHE_DIR / ".lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not (cache.is_file() and cache.stat().st_mtime >= csv_path.stat().st_mtime):
            tx, rx = load_channel_data(csv_path)
            array = np.column_stack((tx.real, tx.imag, rx.real, rx.imag)).astype(np.float64)
            fd, tmp = tempfile.mkstemp(prefix=f".{cache.name}.", suffix=".npy", dir=CACHE_DIR)
            os.close(fd)
            try:
                np.save(tmp, array)
                os.replace(tmp, cache)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return np.load(cache, mmap_mode="r")


@dataclass
class FileSet:
    """Symbols of several files: rx windows are gathered lazily."""

    names: list[str]
    rx: torch.Tensor        # (F, L, 2) float32 network input
    tx: torch.Tensor        # (F, L, 2) float32 regression target
    rx_c: np.ndarray        # (F, L) complex128, for raw BER
    tx_c: np.ndarray        # (F, L) complex128, for BER reference

    @property
    def n_files(self) -> int:
        return self.rx.shape[0]

    @property
    def length(self) -> int:
        return self.rx.shape[1]

    def n_windows(self, window: int) -> int:
        return self.n_files * (self.length - window + 1)

    def to(self, device: torch.device) -> "FileSet":
        return FileSet(self.names, self.rx.to(device), self.tx.to(device), self.rx_c, self.tx_c)

    def gather(self, indices: torch.Tensor, window: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Windows ``(B, W, 2)`` and centre targets ``(B, 2)`` for global indices."""
        per_file = self.length - window + 1
        indices = indices.to(self.rx.device)
        file_ix = torch.div(indices, per_file, rounding_mode="floor")
        offset = indices - file_ix * per_file
        span = torch.arange(window, device=self.rx.device)
        x = self.rx[file_ix[:, None], offset[:, None] + span[None, :]]
        y = self.tx[file_ix, offset + window // 2]
        return x, y

    def centers(self, window: int) -> tuple[np.ndarray, np.ndarray]:
        edge = window // 2
        stop = self.length - edge
        return self.rx_c[:, edge:stop].reshape(-1), self.tx_c[:, edge:stop].reshape(-1)

    def raw_ber(self, window: int) -> float:
        rx_c, tx_c = self.centers(window)
        return ber_from_iq(np.column_stack((rx_c.real, rx_c.imag)), tx_c)


def load_fileset(paths: list[Path]) -> FileSet:
    arrays = [np.asarray(load_symbols(path)) for path in paths]
    for path, array in zip(paths, arrays):
        if array.shape != (SEQUENCE_LENGTH, 4):
            raise ValueError(f"{path}: expected shape {(SEQUENCE_LENGTH, 4)}, got {array.shape}")
    stacked = np.stack(arrays)
    tx_c = stacked[..., 0] + 1j * stacked[..., 1]
    rx_c = stacked[..., 2] + 1j * stacked[..., 3]
    return FileSet(
        names=[path.stem for path in paths],
        rx=torch.from_numpy(stacked[..., 2:4].astype(np.float32)),
        tx=torch.from_numpy(stacked[..., 0:2].astype(np.float32)),
        rx_c=rx_c,
        tx_c=tx_c,
    )


def synthetic_fileset(n_files: int, seed: int, length: int = SEQUENCE_LENGTH) -> FileSet:
    """Pipeline-test data only: 16-QAM, short ISI, Kerr-like phase, AWGN. Never written to results/."""
    from .core import CONSTELLATION

    rng = np.random.default_rng(seed)
    tx_c = CONSTELLATION[rng.integers(0, 16, size=(n_files, length))]
    taps = np.array([0.05 + 0.02j, 1.0, 0.08 - 0.03j])
    rx_c = np.stack([np.convolve(row, taps, mode="same") for row in tx_c])
    power = np.convolve(np.abs(rx_c[0]) ** 2, np.ones(5) / 5, mode="same").mean()
    rx_c = rx_c * np.exp(1j * 0.35 * np.abs(rx_c) ** 2 / power)
    rx_c = rx_c + (rng.normal(size=rx_c.shape) + 1j * rng.normal(size=rx_c.shape)) * 0.09
    rx_c = rx_c / np.sqrt(np.mean(np.abs(rx_c) ** 2))
    rx = np.stack((rx_c.real, rx_c.imag), axis=-1).astype(np.float32)
    tx = np.stack((tx_c.real, tx_c.imag), axis=-1).astype(np.float32)
    return FileSet([f"synthetic_{seed}_{i}" for i in range(n_files)], torch.from_numpy(rx), torch.from_numpy(tx), rx_c, tx_c)


@torch.inference_mode()
def predict_iq(model: torch.nn.Module, data: FileSet, window: int, *, batch_size: int = 8192) -> np.ndarray:
    """Model I/Q predictions for every window, in ``collect`` row order."""
    model.eval()
    out = []
    for f in range(data.n_files):
        windows = data.rx[f].unfold(0, window, 1).transpose(1, 2)  # (L-W+1, W, 2) view
        for start in range(0, windows.shape[0], batch_size):
            prediction = model(windows[start : start + batch_size])
            if torch.is_complex(prediction):
                prediction = torch.stack((prediction.real, prediction.imag), dim=-1)
            out.append(prediction.float().cpu().numpy())
    return np.concatenate(out)


def model_ber(model: torch.nn.Module, data: FileSet, window: int) -> float:
    _, tx_c = data.centers(window)
    return ber_from_iq(predict_iq(model, data, window), tx_c)
