"""Profiling helpers for VSPM and DLPCM experiment preparation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

try:
    import torch
except Exception:  # pragma: no cover - allows metadata use without torch.
    torch = None  # type: ignore


@dataclass(frozen=True)
class TensorStateShape:
    channels: int
    height: int
    width: int
    tensors_per_agent: int = 2
    bytes_per_value: int = 4

    def bytes_per_agent(self) -> int:
        return self.channels * self.height * self.width * self.tensors_per_agent * self.bytes_per_value

    def mib_per_agent(self) -> float:
        return self.bytes_per_agent() / (1024 * 1024)


def count_parameters(model: object, trainable_only: bool = False) -> int:
    """Count parameters for a PyTorch-like module."""

    params = model.parameters()  # type: ignore[attr-defined]
    if trainable_only:
        return sum(p.numel() for p in params if getattr(p, "requires_grad", False))
    return sum(p.numel() for p in params)


def estimate_cache_bytes(num_agents: int, state_shape: TensorStateShape, model_bytes: int = 0) -> int:
    """Estimate receiver cache bytes for model parameters plus active states."""

    if num_agents < 0:
        raise ValueError("num_agents must be non-negative")
    return num_agents * (state_shape.bytes_per_agent() + model_bytes)


def benchmark_callable(
    fn: Callable[[], object],
    warmup: int = 5,
    repeat: int = 20,
    synchronize_cuda: bool = True,
) -> dict[str, float]:
    """Benchmark a callable and return latency statistics in milliseconds."""

    if warmup < 0 or repeat <= 0:
        raise ValueError("warmup must be >= 0 and repeat must be > 0")
    for _ in range(warmup):
        fn()
    if torch is not None and synchronize_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        if torch is not None and synchronize_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    return {
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }

