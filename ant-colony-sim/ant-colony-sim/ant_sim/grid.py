"""GridWorld — 2D cell grid with pheromones, food, obstacles."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .models import (
    FoodSource, PheromoneCell, PheromoneType, Position, NeighborInfo,
)


@dataclass
class Cell:
    x: int
    y: int
    capacity: int  # 0 = obstacle
    is_nest: bool = False
    food_amount: int = 0
    agent_ids: list[int] = field(default_factory=list)

    @property
    def is_obstacle(self) -> bool:
        return self.capacity == 0

    @property
    def has_food(self) -> bool:
        return self.food_amount > 0


class GridWorld:
    def __init__(
        self,
        width: int,
        height: int,
        neighborhood: int = 8,
        default_capacity: int = 99,
        obstacles: list[list[int]] | None = None,
        nest: dict | None = None,
        food_sources: list[dict] | None = None,
        pheromone_cfg: dict | None = None,
        warmstart: dict | None = None,
    ):
        self.width = width
        self.height = height
        self.neighborhood = neighborhood  # 4 or 8

        # Pheromone config
        pcfg = pheromone_cfg or {}
        self.drop_strength: float = pcfg.get("drop_strength", 10)
        self.min_drop_floor: float = pcfg.get("min_drop_floor", 1)
        self.evaporation_rate: float = pcfg.get("evaporation_rate", 0.1)
        self.evaporation_mode: str = pcfg.get("evaporation_mode", "exponential")
        self.round_off_threshold: float = pcfg.get("round_off_threshold", 0.1)

        # Build cells
        self.cells: list[list[Cell]] = [
            [Cell(x=x, y=y, capacity=default_capacity) for x in range(width)]
            for y in range(height)
        ]

        # Place obstacles
        for obs in (obstacles or []):
            ox, oy = obs[0], obs[1]
            if 0 <= oy < height and 0 <= ox < width:
                self.cells[oy][ox].capacity = 0

        # Place nest
        self.nest_x = 0
        self.nest_y = 0
        if nest:
            self.nest_x = nest["x"]
            self.nest_y = nest["y"]
            if 0 <= self.nest_y < height and 0 <= self.nest_x < width:
                self.cells[self.nest_y][self.nest_x].is_nest = True

        # Place food + maintain sparse cache
        self.food_sources_cfg = food_sources or []
        self._food_cells: set[tuple[int, int]] = set()
        self._food_remaining: int = 0
        for fs in self.food_sources_cfg:
            fx, fy = fs["x"], fs["y"]
            if 0 <= fy < height and 0 <= fx < width:
                self.cells[fy][fx].food_amount = fs["amount"]
                self._food_cells.add((fx, fy))
                self._food_remaining += fs["amount"]

        # Pheromone grid (sparse dict for perf)
        self.pheromones: dict[tuple[int, int], PheromoneCell] = {}

        # Warmstart pheromones
        ws = warmstart or {}
        if ws.get("enabled"):
            for trail in ws.get("pheromone_trails", []):
                key = (trail["x"], trail["y"])
                pc = self.pheromones.setdefault(key, PheromoneCell())
                if trail.get("type") == "FOOD":
                    pc.food = trail.get("strength", 1.0)
                else:
                    pc.nest = trail.get("strength", 1.0)

        # Nest food counter
        self.nest_food_total: int = 0

    def cell(self, x: int, y: int) -> Cell:
        return self.cells[y][x]

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_accessible(self, x: int, y: int) -> bool:
        if not self.in_bounds(x, y):
            return False
        return not self.cells[y][x].is_obstacle

    def get_neighbor_positions(self, x: int, y: int) -> list[tuple[int, int]]:
        dirs_8 = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
        dirs_4 = [(0, -1), (-1, 0), (1, 0), (0, 1)]
        dirs = dirs_8 if self.neighborhood == 8 else dirs_4
        result = []
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if self.in_bounds(nx, ny):
                result.append((nx, ny))
        return result

    def get_pheromone(self, x: int, y: int) -> PheromoneCell:
        return self.pheromones.get((x, y), PheromoneCell())

    def deposit_pheromone(self, x: int, y: int, ptype: PheromoneType, strength: float) -> None:
        pc = self.pheromones.setdefault((x, y), PheromoneCell())
        if ptype == PheromoneType.FOOD:
            pc.food += strength
        else:
            pc.nest += strength

    def evaporate_pheromones(self) -> None:
        to_remove = []
        for key, pc in self.pheromones.items():
            if self.evaporation_mode == "exponential":
                pc.food *= (1 - self.evaporation_rate)
                pc.nest *= (1 - self.evaporation_rate)
            else:
                pc.food = max(0, pc.food - self.evaporation_rate)
                pc.nest = max(0, pc.nest - self.evaporation_rate)

            pc.food = math.floor(pc.food * 100) / 100
            pc.nest = math.floor(pc.nest * 100) / 100

            if pc.food < self.round_off_threshold:
                pc.food = 0
            if pc.nest < self.round_off_threshold:
                pc.nest = 0

            if pc.food == 0 and pc.nest == 0:
                to_remove.append(key)

        for key in to_remove:
            del self.pheromones[key]

    def pickup_food_at(self, x: int, y: int) -> bool:
        """Decrement food at (x,y) by 1. Maintains caches. Returns True on success."""
        c = self.cells[y][x]
        if c.food_amount <= 0:
            return False
        c.food_amount -= 1
        self._food_remaining -= 1
        if c.food_amount == 0:
            self._food_cells.discard((x, y))
        return True

    def get_neighbor_info(self, x: int, y: int) -> list[NeighborInfo]:
        result = []
        for nx, ny in self.get_neighbor_positions(x, y):
            c = self.cells[ny][nx]
            pc = self.get_pheromone(nx, ny)
            result.append(NeighborInfo(
                x=nx, y=ny,
                food_pheromone=pc.food,
                nest_pheromone=pc.nest,
                has_food=c.has_food,
                food_amount=c.food_amount,
                is_nest=c.is_nest,
                is_accessible=not c.is_obstacle,
                agent_count=len(c.agent_ids),
            ))
        return result

    def get_sparse_pheromones(self) -> list[dict]:
        result = []
        for (x, y), pc in self.pheromones.items():
            if pc.food > 0 or pc.nest > 0:
                result.append({"x": x, "y": y, "food": round(pc.food, 2), "nest": round(pc.nest, 2)})
        return result

    def get_food_state(self) -> list[dict]:
        result = []
        for (x, y) in self._food_cells:
            result.append({"x": x, "y": y, "amount": self.cells[y][x].food_amount})
        return result

    def total_food_remaining(self) -> int:
        return self._food_remaining

    def get_obstacles(self) -> list[list[int]]:
        result = []
        for row in self.cells:
            for c in row:
                if c.is_obstacle:
                    result.append([c.x, c.y])
        return result
