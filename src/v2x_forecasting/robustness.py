"""Robustness perturbations for revision experiments.

These helpers cover non-ideal communication and sensing conditions requested by
the reviewers: packet loss, delay jitter, BEV degradation, and pose noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PoseNoiseConfig:
    translation_std_m: float = 0.0
    rotation_std_deg: float = 0.0
    translation_bias_m: tuple[float, float] = (0.0, 0.0)
    rotation_bias_deg: float = 0.0


def packet_keep_mask(num_packets: int, loss_rate: float, seed: int | None = None) -> np.ndarray:
    """Return a boolean mask where True means the packet is retained."""

    if not 0.0 <= loss_rate <= 1.0:
        raise ValueError("loss_rate must be in [0, 1]")
    rng = np.random.default_rng(seed)
    return rng.random(num_packets) >= loss_rate


def apply_packet_loss(items: Sequence[object], loss_rate: float, seed: int | None = None) -> list[object]:
    """Drop sequence items according to an independent Bernoulli loss rate."""

    mask = packet_keep_mask(len(items), loss_rate, seed)
    return [item for item, keep in zip(items, mask) if bool(keep)]


def jittered_delays(
    base_delay_frames: int,
    count: int,
    max_abs_jitter_frames: int,
    seed: int | None = None,
) -> np.ndarray:
    """Sample integer delay jitter around a base delay."""

    if base_delay_frames < 0:
        raise ValueError("base_delay_frames must be non-negative")
    if max_abs_jitter_frames < 0:
        raise ValueError("max_abs_jitter_frames must be non-negative")
    rng = np.random.default_rng(seed)
    jitter = rng.integers(-max_abs_jitter_frames, max_abs_jitter_frames + 1, size=count)
    return np.maximum(0, base_delay_frames + jitter)


def degrade_bev(
    bev: np.ndarray,
    dropout_rate: float = 0.0,
    false_positive_rate: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Apply simple occupancy dropout and false positives to a BEV grid."""

    if not 0.0 <= dropout_rate <= 1.0:
        raise ValueError("dropout_rate must be in [0, 1]")
    if not 0.0 <= false_positive_rate <= 1.0:
        raise ValueError("false_positive_rate must be in [0, 1]")
    rng = np.random.default_rng(seed)
    out = (bev > 0).astype(np.uint8, copy=True)
    if dropout_rate > 0:
        drop = (out == 1) & (rng.random(out.shape) < dropout_rate)
        out[drop] = 0
    if false_positive_rate > 0:
        add = (out == 0) & (rng.random(out.shape) < false_positive_rate)
        out[add] = 1
    return out


def sample_pose_noise(config: PoseNoiseConfig, seed: int | None = None) -> tuple[float, float, float]:
    """Sample dx, dy, dtheta perturbations for spatial-alignment experiments."""

    rng = np.random.default_rng(seed)
    dx = rng.normal(config.translation_bias_m[0], config.translation_std_m)
    dy = rng.normal(config.translation_bias_m[1], config.translation_std_m)
    dtheta = rng.normal(config.rotation_bias_deg, config.rotation_std_deg)
    return float(dx), float(dy), float(dtheta)

