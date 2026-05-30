"""Utilities for frame-level communication/computation latency simulation.

The functions here are deliberately framework-agnostic. They prepare the timing
indices used by later VSPM/DLPCM experiments; model-specific rollout code can
consume the returned horizons without depending on a detection pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class LatencyCase:
    """A discrete dual-latency setting in frames."""

    comm_delay: int
    comp_delay: int
    frame_interval_s: float = 0.2

    def __post_init__(self) -> None:
        if self.comm_delay < 0:
            raise ValueError("comm_delay must be non-negative")
        if self.comp_delay < 0:
            raise ValueError("comp_delay must be non-negative")
        if self.frame_interval_s <= 0:
            raise ValueError("frame_interval_s must be positive")

    @property
    def remote_rollout_frames(self) -> int:
        """Frames needed to align a remote packet to final result time."""

        return self.comm_delay + self.comp_delay

    @property
    def local_rollout_frames(self) -> int:
        """Frames needed to align local ego perception to final result time."""

        return self.comp_delay

    @property
    def comm_delay_s(self) -> float:
        return self.comm_delay * self.frame_interval_s

    @property
    def comp_delay_s(self) -> float:
        return self.comp_delay * self.frame_interval_s


def make_latency_grid(
    comm_delays: Iterable[int],
    comp_delays: Iterable[int],
    frame_interval_s: float = 0.2,
) -> list[LatencyCase]:
    """Create a deterministic grid of latency settings."""

    return [
        LatencyCase(int(comm), int(comp), frame_interval_s)
        for comp in comp_delays
        for comm in comm_delays
    ]


def select_delayed_index(current_index: int, delay_frames: int) -> int:
    """Return the source frame index observed after a delay."""

    if delay_frames < 0:
        raise ValueError("delay_frames must be non-negative")
    return max(0, current_index - delay_frames)


def shifted_sequence_indices(length: int, delay_frames: int) -> list[int]:
    """Map each current frame to the delayed source frame index."""

    if length < 0:
        raise ValueError("length must be non-negative")
    return [select_delayed_index(i, delay_frames) for i in range(length)]


def future_target_index(current_index: int, comp_delay_frames: int) -> int:
    """Final output target index after computation delay."""

    if comp_delay_frames < 0:
        raise ValueError("comp_delay_frames must be non-negative")
    return current_index + comp_delay_frames


def rollout_horizons_for_agents(
    sender_frame_indices: Sequence[int],
    result_frame_index: int,
) -> list[int]:
    """Compute per-agent rollout horizons to a shared result frame."""

    return [max(0, result_frame_index - int(idx)) for idx in sender_frame_indices]

