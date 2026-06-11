"""SimQueue — max 4 concurrent sims + pending queue."""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .engine import SimulationEngine
from .tick_store import TickStore


class SimStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SimEntry:
    id: str
    name: str
    config: dict
    status: SimStatus = SimStatus.PENDING
    engine: SimulationEngine | None = None
    store: TickStore | None = None
    agent_class: type | None = None
    agent_code: str = ""
    tick_states: list[dict] = field(default_factory=list)
    metrics: dict | None = None
    error: str | None = None


class SimQueue:
    MAX_CONCURRENT = 4

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.entries: dict[str, SimEntry] = {}
        self._running: set[str] = set()
        self._pending: list[str] = []

    def add(self, config: dict, agent_class: type | None = None,
            agent_code: str = "") -> str:
        sim_id = uuid.uuid4().hex[:12]
        name = config.get("name", sim_id)
        entry = SimEntry(
            id=sim_id,
            name=name,
            config=config,
            agent_class=agent_class,
            agent_code=agent_code,
        )
        self.entries[sim_id] = entry
        self._pending.append(sim_id)
        return sim_id

    async def run_next(self) -> str | None:
        """Start next pending sim if slot available. Returns sim_id or None."""
        if len(self._running) >= self.MAX_CONCURRENT:
            return None
        if not self._pending:
            return None

        sim_id = self._pending.pop(0)
        entry = self.entries[sim_id]
        entry.status = SimStatus.RUNNING
        self._running.add(sim_id)

        # Setup store
        sim_dir = self.data_dir / entry.name
        if sim_dir.exists():
            entry.name += f"_{uuid.uuid4().hex[:12]}"
            sim_dir = self.data_dir / entry.name
            self.entries[sim_id] = entry
        sim_dir.mkdir(parents=True, exist_ok=True)
        entry.store = TickStore(sim_dir)
        entry.store.save_config(entry.config)
        if entry.agent_code:
            entry.store.save_agent_code(entry.agent_code)

        # Run simulation in executor
        try:
            await self._run_simulation(entry)
        except Exception as e:
            entry.status = SimStatus.FAILED
            entry.error = str(e)
        finally:
            self._running.discard(sim_id)

        # Start next if available
        asyncio.create_task(self._try_start_next())

        return sim_id

    async def _try_start_next(self) -> None:
        if self._pending and len(self._running) < self.MAX_CONCURRENT:
            await self.run_next()

    async def _run_simulation(self, entry: SimEntry) -> None:
        engine = SimulationEngine(entry.config, entry.agent_class)
        entry.engine = engine

        loop = asyncio.get_event_loop()

        def _run():
            gen = engine.run()
            for ts in gen:
                td = ts.to_dict()
                entry.tick_states.append(td)  # live update, SSE-stream sieht es sofort
                if entry.store:
                    entry.store.append(td)
                    if ts.tick % engine.snapshot_interval == 0:
                        entry.store.save_snapshot(ts.tick, td)

        await loop.run_in_executor(None, _run)
        entry.metrics = engine.get_metrics().to_dict()
        entry.status = SimStatus.COMPLETED

        if entry.store:
            entry.store.flush()
            if entry.metrics:
                entry.store.save_metrics(entry.metrics)

    def load_from_disk(self, name: str) -> SimEntry | None:
        """Lazy-load a completed sim from disk into entries."""
        sim_dir = self.data_dir / name
        if not sim_dir.is_dir():
            return None
        cfg_path = sim_dir / "config.json"
        if not cfg_path.exists():
            return None

        import json
        try:
            cfg = json.loads(cfg_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

        store = TickStore(sim_dir)
        entry = SimEntry(
            id=name,
            name=cfg.get("name", name),
            config=cfg,
            status=SimStatus.COMPLETED,
            store=store,
        )

        metrics_path = sim_dir / "metrics.json"
        if metrics_path.exists():
            try:
                entry.metrics = json.loads(metrics_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        self.entries[name] = entry
        return entry

    def get_status(self, sim_id: str) -> dict | None:
        entry = self.entries.get(sim_id)
        if not entry:
            return None
        return {
            "id": entry.id,
            "name": entry.name,
            "status": entry.status.value,
            "ticks": len(entry.tick_states),
            "error": entry.error,
        }

    def list_all(self) -> list[dict]:
        result = []
        for e in self.entries.values():
            result.append({
                "id": e.id,
                "name": e.name,
                "status": e.status.value,
                "ticks": len(e.tick_states),
                "has_metrics": e.metrics is not None,
            })
        return result

    def pause(self, sim_id: str) -> bool:
        entry = self.entries.get(sim_id)
        if entry and entry.engine:
            entry.engine.paused = True
            entry.status = SimStatus.PAUSED
            return True
        return False

    def get_queue_info(self) -> dict:
        return {
            "running": list(self._running),
            "pending": list(self._pending),
            "running_count": len(self._running),
            "pending_count": len(self._pending),
            "max_concurrent": self.MAX_CONCURRENT,
        }
