"""Domain models — 1:1 from class diagram + spec v3.0."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Generator, Union


# --- Enums ---

class PheromoneType(Enum):
    FOOD = "FOOD"
    NEST = "NEST"


class ActionResult(Enum):
    SUCCESS = "success"
    ILLEGAL = "illegal"
    NO_ENERGY = "no_energy"


# --- Value Objects ---

@dataclass(frozen=True)
class Position:
    x: int
    y: int

    def distance_to(self, other: Position) -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


# --- Actions (Agent → Manager) ---

@dataclass(frozen=True)
class MoveAction:
    target_x: int
    target_y: int


@dataclass(frozen=True)
class PickUpAction:
    source_x: int
    source_y: int


@dataclass(frozen=True)
class DropAction:
    target_x: int
    target_y: int


Action = Union[MoveAction, PickUpAction, DropAction]


# --- Items ---

@dataclass
class FoodSource:
    x: int
    y: int
    amount: int


# --- Pheromone grid (sparse) ---

@dataclass
class PheromoneCell:
    food: float = 0.0
    nest: float = 0.0


# --- Perception (Manager → Agent) ---

@dataclass
class NeighborInfo:
    x: int
    y: int
    food_pheromone: float
    nest_pheromone: float
    has_food: bool
    food_amount: int
    is_nest: bool
    is_accessible: bool
    agent_count: int


@dataclass
class Perception:
    neighbors: list[NeighborInfo]
    current_x: int
    current_y: int
    current_food_pheromone: float
    current_nest_pheromone: float
    current_has_food: bool
    current_is_nest: bool
    last_action_result: ActionResult | None
    energy: int
    carries: bool
    tick: int


# --- Agent ---

@dataclass
class AntAgent:
    id: int
    energy: int
    carries: bool = False
    x: int = 0
    y: int = 0
    memory: list[Position] = field(default_factory=list)
    memory_size: int = 5
    alive: bool = True
    last_action: str | None = None
    last_action_result: ActionResult | None = None
    steps_since_source: int = 0

    def push_memory(self, pos: Position) -> None:
        self.memory.append(pos)
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def decide(self, perception: Perception) -> Action:
        raise NotImplementedError


# --- Tick State (serializable) ---

@dataclass
class AgentSnapshot:
    id: int
    x: int
    y: int
    energy: int
    carries: bool
    alive: bool
    action: str | None
    action_result: str | None


@dataclass
class TickState:
    tick: int
    pheromones: list[dict]
    food_sources: list[dict]
    agents: list[AgentSnapshot]
    nest_food_total: int
    alive_count: int
    dead_count: int

    def to_dict(self) -> dict:
        return {
            "tick": self.tick,
            "pheromones": self.pheromones,
            "food_sources": self.food_sources,
            "agents": [
                {
                    "id": a.id, "x": a.x, "y": a.y,
                    "energy": a.energy, "carries": a.carries,
                    "alive": a.alive, "action": a.action,
                    "action_result": a.action_result,
                }
                for a in self.agents
            ],
            "nest_food_total": self.nest_food_total,
            "alive_count": self.alive_count,
            "dead_count": self.dead_count,
        }


# --- Metrics ---

@dataclass
class TripRecord:
    agent_id: int
    t_leave_nest: int
    t_find_food: int
    t_return_nest: int
    steps_to_food: int


@dataclass
class PeriodicSnapshot:
    tick: int
    food_collected_total: int
    alive_count: int
    dead_count: int
    avg_steps_to_food: float


@dataclass
class SimulationMetrics:
    name: str
    config_hash: str
    total_ticks: int
    total_ants: int
    food_collected_total: int
    food_per_ant: float
    food_per_ant_per_step: float
    avg_steps_to_food: float
    trips: list[TripRecord]
    alive_ants: int
    dead_ants: int
    death_ratio: float
    snapshots: list[PeriodicSnapshot]
    termination_reason: str
    completed: bool

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "config_hash": self.config_hash,
            "total_ticks": self.total_ticks,
            "total_ants": self.total_ants,
            "food_collected_total": self.food_collected_total,
            "food_per_ant": self.food_per_ant,
            "food_per_ant_per_step": self.food_per_ant_per_step,
            "avg_steps_to_food": self.avg_steps_to_food,
            "trips": [
                {
                    "agent_id": t.agent_id,
                    "t_leave_nest": t.t_leave_nest,
                    "t_find_food": t.t_find_food,
                    "t_return_nest": t.t_return_nest,
                    "steps_to_food": t.steps_to_food,
                }
                for t in self.trips
            ],
            "alive_ants": self.alive_ants,
            "dead_ants": self.dead_ants,
            "death_ratio": self.death_ratio,
            "snapshots": [
                {
                    "tick": s.tick,
                    "food_collected_total": s.food_collected_total,
                    "alive_count": s.alive_count,
                    "dead_count": s.dead_count,
                    "avg_steps_to_food": s.avg_steps_to_food,
                }
                for s in self.snapshots
            ],
            "termination_reason": self.termination_reason,
            "completed": self.completed,
        }
