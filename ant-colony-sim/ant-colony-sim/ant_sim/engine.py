"""SimulationManager — headless engine, yields TickState per tick."""
from __future__ import annotations

import hashlib
import json
import random
import signal
import time
from contextlib import contextmanager
from typing import Generator

from .grid import GridWorld
from .models import (
    Action, ActionResult, AgentSnapshot, AntAgent, DropAction, FoodSource,
    MoveAction, Perception, PheromoneType, PickUpAction, Position,
    PeriodicSnapshot, SimulationMetrics, TickState, TripRecord,
)


@contextmanager
def timeout_ctx(ms: int):
    """Timeout context manager (unix only, fallback = no timeout)."""
    if ms <= 0:
        yield
        return
    try:
        def _handler(signum, frame):
            raise TimeoutError()
        old = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, ms / 1000.0)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
    except (AttributeError, ValueError):
        # Windows or no signal support
        yield


class SimulationEngine:
    def __init__(self, config: dict, agent_class: type | None = None):
        self.config = config
        self.name = config.get("name", "unnamed")
        self.config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]

        grid_cfg = config.get("grid", {})
        self.grid = GridWorld(
            width=grid_cfg.get("width", 20),
            height=grid_cfg.get("height", 20),
            neighborhood=grid_cfg.get("neighborhood", 8),
            default_capacity=grid_cfg.get("default_capacity", 99),
            obstacles=grid_cfg.get("obstacles"),
            nest=config.get("nest"),
            food_sources=config.get("food_sources"),
            pheromone_cfg=config.get("pheromones"),
            warmstart=config.get("warmstart"),
        )

        agents_cfg = config.get("agents", {})
        self.agent_count = agents_cfg.get("count", 10)
        self.initial_energy = agents_cfg.get("initial_energy", 200)
        self.memory_size = agents_cfg.get("memory_size", 5)

        sim_cfg = config.get("simulation", {})
        self.max_ticks = sim_cfg.get("max_ticks", 1000)
        self.random_seed = sim_cfg.get("random_seed", 42)
        self.termination = sim_cfg.get("termination", ["max_ticks"])
        self.snapshot_interval = sim_cfg.get("snapshot_interval", 5)
        self.agent_timeout_ms = sim_cfg.get("agent_timeout_ms", 100)

        phero_cfg = config.get("pheromones", {})
        self.drop_strength = phero_cfg.get("drop_strength", 10)
        self.min_drop_floor = phero_cfg.get("min_drop_floor", 1)

        # Agent class (default or user-provided)
        if agent_class is None:
            from agents.default_agent import DefaultAgent
            self.agent_class = DefaultAgent
        else:
            self.agent_class = agent_class

        # State
        self.agents: list[AntAgent] = []
        self.tick = 0
        self.nest_food_total = 0
        self.trips: list[TripRecord] = []
        self.snapshots: list[PeriodicSnapshot] = []
        self.termination_reason = ""
        self.completed = False
        self.paused = False

        # Trip tracking per agent
        self._trip_leave: dict[int, int] = {}
        self._trip_find: dict[int, int] = {}

    def initialize(self) -> None:
        random.seed(self.random_seed)
        nest_x = self.grid.nest_x
        nest_y = self.grid.nest_y

        for i in range(self.agent_count):
            agent = self.agent_class(
                id=i, energy=self.initial_energy,
                x=nest_x, y=nest_y, memory_size=self.memory_size,
            )
            self.agents.append(agent)
            self.grid.cell(nest_x, nest_y).agent_ids.append(i)

    def _build_perception(self, agent: AntAgent) -> Perception:
        pc = self.grid.get_pheromone(agent.x, agent.y)
        c = self.grid.cell(agent.x, agent.y)
        return Perception(
            neighbors=self.grid.get_neighbor_info(agent.x, agent.y),
            current_x=agent.x,
            current_y=agent.y,
            current_food_pheromone=pc.food,
            current_nest_pheromone=pc.nest,
            current_has_food=c.has_food,
            current_is_nest=c.is_nest,
            last_action_result=agent.last_action_result,
            energy=agent.energy,
            carries=agent.carries,
            tick=self.tick,
        )

    def _execute_action(self, agent: AntAgent, action: Action) -> ActionResult:
        if agent.energy <= 0:
            return ActionResult.NO_ENERGY

        if isinstance(action, MoveAction):
            tx, ty = action.target_x, action.target_y
            # Validate: must be neighbor and accessible
            neighbors = self.grid.get_neighbor_positions(agent.x, agent.y)
            if (tx, ty) not in neighbors or not self.grid.is_accessible(tx, ty):
                agent.last_action = "move"
                return ActionResult.ILLEGAL

            # Move
            old_cell = self.grid.cell(agent.x, agent.y)
            was_on_nest = old_cell.is_nest
            self.grid.cell(agent.x, agent.y).agent_ids.remove(agent.id)
            agent.x = tx
            agent.y = ty
            self.grid.cell(tx, ty).agent_ids.append(agent.id)
            agent.push_memory(Position(tx, ty))
            agent.last_action = "move"
            agent.energy -= 1
            agent.steps_since_source += 1
            # Track nest departure
            if was_on_nest and not self.grid.cell(tx, ty).is_nest:
                self._trip_leave[agent.id] = self.tick
            return ActionResult.SUCCESS

        elif isinstance(action, PickUpAction):
            c = self.grid.cell(agent.x, agent.y)
            if not c.has_food or agent.carries:
                agent.last_action = "pickup"
                return ActionResult.ILLEGAL
            self.grid.pickup_food_at(agent.x, agent.y)
            agent.carries = True
            agent.last_action = "pickup"
            agent.energy -= 1
            agent.steps_since_source = 0
            # Trip tracking
            self._trip_find[agent.id] = self.tick
            return ActionResult.SUCCESS

        elif isinstance(action, DropAction):
            c = self.grid.cell(agent.x, agent.y)
            if not c.is_nest or not agent.carries:
                agent.last_action = "drop"
                return ActionResult.ILLEGAL
            agent.carries = False
            self.nest_food_total += 1
            agent.last_action = "drop"
            agent.energy -= 1
            agent.steps_since_source = 0
            # Complete trip (only if both departure and find were tracked)
            if agent.id in self._trip_find and agent.id in self._trip_leave:
                self.trips.append(TripRecord(
                    agent_id=agent.id,
                    t_leave_nest=self._trip_leave[agent.id],
                    t_find_food=self._trip_find[agent.id],
                    t_return_nest=self.tick,
                    steps_to_food=self._trip_find[agent.id] - self._trip_leave[agent.id],
                ))
                del self._trip_find[agent.id]
                del self._trip_leave[agent.id]
            return ActionResult.SUCCESS

        return ActionResult.ILLEGAL

    def _deposit_pheromone(self, agent: AntAgent) -> None:
        if not agent.alive:
            return
        ptype = PheromoneType.FOOD if agent.carries else PheromoneType.NEST
        strength = max(self.min_drop_floor, self.drop_strength - agent.steps_since_source)
        self.grid.deposit_pheromone(agent.x, agent.y, ptype, strength)

    def _energy_refill(self, agent: AntAgent) -> None:
        c = self.grid.cell(agent.x, agent.y)
        if c.is_nest:
            agent.energy = self.initial_energy
            agent.steps_since_source = 0

    def _check_death(self, agent: AntAgent) -> None:
        if agent.energy <= 0 and agent.alive:
            agent.alive = False
            if agent.id in self.grid.cell(agent.x, agent.y).agent_ids:
                self.grid.cell(agent.x, agent.y).agent_ids.remove(agent.id)

    def _build_tick_state(self) -> TickState:
        alive = sum(1 for a in self.agents if a.alive)
        dead = len(self.agents) - alive
        return TickState(
            tick=self.tick,
            pheromones=self.grid.get_sparse_pheromones(),
            food_sources=self.grid.get_food_state(),
            agents=[
                AgentSnapshot(
                    id=a.id, x=a.x, y=a.y,
                    energy=a.energy, carries=a.carries,
                    alive=a.alive, action=a.last_action,
                    action_result=a.last_action_result.value if a.last_action_result else None,
                )
                for a in self.agents
            ],
            nest_food_total=self.nest_food_total,
            alive_count=alive,
            dead_count=dead,
        )

    def _check_termination(self) -> str | None:
        if "max_ticks" in self.termination and self.tick >= self.max_ticks:
            return "max_ticks"
        if "no_food_present" in self.termination and self.grid.total_food_remaining() == 0:
            # Also check no agent is carrying
            if not any(a.carries for a in self.agents if a.alive):
                return "no_food_present"
        if "all_agents_dead" in self.termination:
            if all(not a.alive for a in self.agents):
                return "all_agents_dead"
        return None

    def run(self) -> Generator[TickState, None, SimulationMetrics]:
        """Generator yielding TickState per tick. Returns SimulationMetrics."""
        self.initialize()

        # Yield initial state (tick 0)
        yield self._build_tick_state()

        for _ in range(self.max_ticks):
            if self.paused:
                return  # type: ignore

            self.tick += 1

            # Evaporate pheromones
            self.grid.evaporate_pheromones()

            # Process each living agent
            for agent in self.agents:
                if not agent.alive:
                    continue

                # Build perception
                perception = self._build_perception(agent)

                # Agent decides
                action: Action | None = None
                try:
                    with timeout_ctx(self.agent_timeout_ms):
                        action = agent.decide(perception)
                except (TimeoutError, NotImplementedError, Exception):
                    agent.last_action = None
                    agent.last_action_result = None
                    agent.energy -= 1
                    continue

                if action is None:
                    agent.last_action = None
                    agent.last_action_result = None
                    agent.energy -= 1
                    continue

                # Execute action
                result = self._execute_action(agent, action)
                agent.last_action_result = result

                # Engine deposits pheromone
                self._deposit_pheromone(agent)

                # Energy refill on nest
                self._energy_refill(agent)

                # Death check
                self._check_death(agent)

            # Build tick state
            ts = self._build_tick_state()

            # Periodic snapshot
            if self.tick % self.snapshot_interval == 0:
                avg_stf = 0.0
                if self.trips:
                    avg_stf = sum(t.steps_to_food for t in self.trips) / len(self.trips)
                self.snapshots.append(PeriodicSnapshot(
                    tick=self.tick,
                    food_collected_total=self.nest_food_total,
                    alive_count=ts.alive_count,
                    dead_count=ts.dead_count,
                    avg_steps_to_food=round(avg_stf, 2),
                ))

            yield ts

            # Check termination
            reason = self._check_termination()
            if reason:
                self.termination_reason = reason
                break

        if not self.termination_reason:
            self.termination_reason = "max_ticks"

        self.completed = True

    def get_metrics(self) -> SimulationMetrics:
        alive = sum(1 for a in self.agents if a.alive)
        dead = len(self.agents) - alive
        avg_stf = 0.0
        if self.trips:
            avg_stf = sum(t.steps_to_food for t in self.trips) / len(self.trips)

        fpa = self.nest_food_total / max(1, len(self.agents))
        fpaps = fpa / max(1, self.tick)

        return SimulationMetrics(
            name=self.name,
            config_hash=self.config_hash,
            total_ticks=self.tick,
            total_ants=len(self.agents),
            food_collected_total=self.nest_food_total,
            food_per_ant=round(fpa, 4),
            food_per_ant_per_step=round(fpaps, 6),
            avg_steps_to_food=round(avg_stf, 2),
            trips=self.trips,
            alive_ants=alive,
            dead_ants=dead,
            death_ratio=round(dead / max(1, len(self.agents)), 4),
            snapshots=self.snapshots,
            termination_reason=self.termination_reason,
            completed=self.completed,
        )
