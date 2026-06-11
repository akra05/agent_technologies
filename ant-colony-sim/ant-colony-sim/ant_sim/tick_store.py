"""TickStore — RAM sliding window + gzip chunks to disk."""
from __future__ import annotations

import gzip
import json
import os
from collections import deque
from pathlib import Path
from typing import Any


CHUNK_SIZE = 100


class TickStore:
    def __init__(self, sim_dir: Path, window_size: int = 20):
        self.sim_dir = sim_dir
        self.chunks_dir = sim_dir / "chunks"
        self.snapshots_dir = sim_dir / "snapshots"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.window_size = window_size

        self._buffer: deque[dict] = deque(maxlen=window_size)
        self._chunk_buffer: list[dict] = []
        self._chunk_index: int = 0
        self._total_ticks: int = 0

    def append(self, tick_data: dict) -> None:
        self._buffer.append(tick_data)
        self._chunk_buffer.append(tick_data)
        self._total_ticks += 1

        if len(self._chunk_buffer) >= CHUNK_SIZE:
            self._flush_chunk()

    def _flush_chunk(self) -> None:
        if not self._chunk_buffer:
            return
        start = self._chunk_index * CHUNK_SIZE
        end = start + len(self._chunk_buffer) - 1
        filename = f"ticks_{start:04d}-{end:04d}.json.gz"
        filepath = self.chunks_dir / filename

        with gzip.open(filepath, "wt", encoding="utf-8") as f:
            json.dump(self._chunk_buffer, f)

        self._chunk_buffer = []
        self._chunk_index += 1

    def flush(self) -> None:
        """Flush remaining buffer to disk."""
        if self._chunk_buffer:
            self._flush_chunk()

    def save_snapshot(self, tick: int, state: dict) -> None:
        filepath = self.snapshots_dir / f"snapshot_{tick:06d}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f)

    def get_tick(self, tick_num: int) -> dict | None:
        # Check RAM window first
        for td in self._buffer:
            if td.get("tick") == tick_num:
                return td

        # Check disk chunks
        chunk_idx = tick_num // CHUNK_SIZE
        start = chunk_idx * CHUNK_SIZE
        end = start + CHUNK_SIZE - 1

        # Try to find matching chunk file
        for f in self.chunks_dir.iterdir():
            if f.name.startswith(f"ticks_{start:04d}"):
                with gzip.open(f, "rt", encoding="utf-8") as fh:
                    ticks = json.load(fh)
                    for td in ticks:
                        if td.get("tick") == tick_num:
                            return td
        return None

    def get_recent(self, n: int = 10) -> list[dict]:
        return list(self._buffer)[-n:]

    @property
    def total_ticks(self) -> int:
        return self._total_ticks

    def save_config(self, config: dict) -> None:
        with open(self.sim_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def save_agent_code(self, code: str) -> None:
        with open(self.sim_dir / "agent.py", "w", encoding="utf-8") as f:
            f.write(code)

    def save_metrics(self, metrics: dict) -> None:
        with open(self.sim_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    def iter_all(self):
        """Yield all ticks in order: flushed chunks from disk, then unflushed buffer."""
        chunk_files = sorted(
            (f for f in self.chunks_dir.iterdir() if f.name.startswith("ticks_")),
            key=lambda f: int(f.name.split("_")[1].split("-")[0])
        )
        for cf in chunk_files:
            with gzip.open(cf, "rt", encoding="utf-8") as fh:
                ticks = json.load(fh)
            yield from ticks

        # unflushed remainder
        yield from self._chunk_buffer

    def iter_chunks_raw(self):
        """Yield pre-serialized JSON array strings, one per chunk.
        Avoids json.load + json.dumps roundtrip — reads gzip text directly."""
        chunk_files = sorted(
            (f for f in self.chunks_dir.iterdir() if f.name.startswith("ticks_")),
            key=lambda f: int(f.name.split("_")[1].split("-")[0])
        )
        for cf in chunk_files:
            with gzip.open(cf, "rt", encoding="utf-8") as fh:
                yield fh.read()  # already a valid JSON array string

        # unflushed remainder
        if self._chunk_buffer:
            yield json.dumps(self._chunk_buffer)
