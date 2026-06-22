"""Receiver-side VSPM cache utilities for scalability experiments."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass
class CacheEntry:
    vehicle_id: str
    value: Any
    last_seen_frame: int
    distance_m: float | None = None


class VSPMCache:
    """Small LRU/TTL/range cache used to model receiver-side VSPM storage."""

    def __init__(
        self,
        capacity: int,
        ttl_frames: int | None = None,
        max_range_m: float | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if ttl_frames is not None and ttl_frames < 0:
            raise ValueError("ttl_frames must be non-negative")
        if max_range_m is not None and max_range_m < 0:
            raise ValueError("max_range_m must be non-negative")
        self.capacity = capacity
        self.ttl_frames = ttl_frames
        self.max_range_m = max_range_m
        self._items: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def __len__(self) -> int:
        return len(self._items)

    def keys(self) -> list[str]:
        return list(self._items.keys())

    def get(self, vehicle_id: str, current_frame: int | None = None) -> Any | None:
        entry = self._items.get(vehicle_id)
        if entry is None:
            self.misses += 1
            return None
        if current_frame is not None and self._expired(entry, current_frame):
            self._items.pop(vehicle_id, None)
            self.evictions += 1
            self.misses += 1
            return None
        self._items.move_to_end(vehicle_id)
        self.hits += 1
        return entry.value

    def put(
        self,
        vehicle_id: str,
        value: Any,
        frame_index: int,
        distance_m: float | None = None,
    ) -> None:
        if vehicle_id in self._items:
            self._items.move_to_end(vehicle_id)
        self._items[vehicle_id] = CacheEntry(vehicle_id, value, frame_index, distance_m)
        self.evict(frame_index)

    def evict(self, current_frame: int) -> list[str]:
        removed: list[str] = []
        for vehicle_id, entry in list(self._items.items()):
            if self._expired(entry, current_frame):
                self._items.pop(vehicle_id, None)
                removed.append(vehicle_id)
        while len(self._items) > self.capacity:
            vehicle_id, _ = self._items.popitem(last=False)
            removed.append(vehicle_id)
        self.evictions += len(removed)
        return removed

    def stats(self) -> dict[str, float | int]:
        total = self.hits + self.misses
        return {
            "size": len(self._items),
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": self.hits / total if total else 0.0,
        }

    def _expired(self, entry: CacheEntry, current_frame: int) -> bool:
        if self.ttl_frames is not None and current_frame - entry.last_seen_frame > self.ttl_frames:
            return True
        if (
            self.max_range_m is not None
            and entry.distance_m is not None
            and entry.distance_m > self.max_range_m
        ):
            return True
        return False

