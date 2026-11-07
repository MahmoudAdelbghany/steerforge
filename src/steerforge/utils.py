from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

T = TypeVar("T")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stable_hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, target)


def chunked(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def flatten(groups: Iterable[Iterable[T]]) -> list[T]:
    return [item for group in groups for item in group]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def human_int(value: int | float | None) -> str:
    if value is None:
        return "?"
    value = float(value)
    for suffix in ("", "K", "M", "B", "T"):
        if abs(value) < 1000 or suffix == "T":
            if suffix:
                return f"{value:.1f}{suffix}".replace(".0", "")
            return f"{int(value)}"
        value /= 1000
    return str(value)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    seconds = float(seconds)
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {seconds:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


def now() -> float:
    return time.perf_counter()


def safe_name(value: str) -> str:
    keep = [c if c.isalnum() or c in "-_." else "-" for c in value.strip().lower()]
    out = "".join(keep).strip("-.")
    return out or "persona"


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Rank values to [0, 1], with ties averaged."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    # average ties
    sorted_values = values[order]
    start = 0
    for end in range(1, values.size + 1):
        if end == values.size or sorted_values[end] != sorted_values[start]:
            if end - start > 1:
                ranks[order[start:end]] = (start + end - 1) / 2.0
            start = end
    return ranks / max(values.size - 1, 1)
