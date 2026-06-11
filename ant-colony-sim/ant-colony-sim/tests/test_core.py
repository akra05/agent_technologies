"""Tests — unittest only."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ant_sim.models import (
    ActionResult, AntAgent, DropAction, MoveAction, Perception, PheromoneType,
    PickUpAction, Position, TickState,
)
from ant_sim.grid import GridWorld
from ant_sim.engine import SimulationEngine
from ant_sim.validation import validate_config, load_agent_class, expand_batch
from ant_sim.metrics import compute_batch_scores


class TestPosition(unittest.TestCase):
    def test_distance(self):
        p1 = Position(0, 0)
        p2 = Position(3, 4)
        self.assertAlmostEqual(p1.distance_to(p2), 5.0)

    def test_frozen(self):
        p = Position(1, 2)
        with self.assertRaises(AttributeError):
            p.x = 5


class TestGridWorld(unittest.TestCase):
    def setUp(self):
        self.grid = GridWorld(
            width=10, height=10, neighborhood=8,
            obstacles=[[3, 3], [3, 4]],
            nest={"x": 5, "y": 5},
            food_sources=[{"x": 1, "y": 1, "amount": 20}],
            pheromone_cfg={"evaporation_rate": 0.1, "round_off_threshold": 0.1},
        )

    def test_dimensions(self):
        self.assertEqual(self.grid.width, 10)
        self.assertEqual(self.grid.height, 10)

    def test_obstacles(self):
        self.assertTrue(self.grid.cell(3, 3).is_obstacle)
        self.assertTrue(self.grid.cell(3, 4).is_obstacle)
        self.assertFalse(self.grid.cell(0, 0).is_obstacle)

    def test_nest(self):
        self.assertTrue(self.grid.cell(5, 5).is_nest)
        self.assertFalse(self.grid.cell(0, 0).is_nest)

    def test_food(self):
        self.assertEqual(self.grid.cell(1, 1).food_amount, 20)
        self.assertTrue(self.grid.cell(1, 1).has_food)

    def test_accessible(self):
        self.assertTrue(self.grid.is_accessible(0, 0))
        self.assertFalse(self.grid.is_accessible(3, 3))
        self.assertFalse(self.grid.is_accessible(-1, 0))
        self.assertFalse(self.grid.is_accessible(10, 0))

    def test_neighbors_8(self):
        neighbors = self.grid.get_neighbor_positions(5, 5)
        self.assertEqual(len(neighbors), 8)

    def test_neighbors_corner(self):
        neighbors = self.grid.get_neighbor_positions(0, 0)
        self.assertEqual(len(neighbors), 3)

    def test_pheromone_deposit_evaporate(self):
        self.grid.deposit_pheromone(2, 2, PheromoneType.FOOD, 10.0)
        pc = self.grid.get_pheromone(2, 2)
        self.assertEqual(pc.food, 10.0)

        self.grid.evaporate_pheromones()
        pc = self.grid.get_pheromone(2, 2)
        self.assertLess(pc.food, 10.0)

    def test_pheromone_round_off(self):
        self.grid.deposit_pheromone(2, 2, PheromoneType.FOOD, 0.05)
        self.grid.evaporate_pheromones()
        pc = self.grid.get_pheromone(2, 2)
        self.assertEqual(pc.food, 0)

    def test_sparse_pheromones(self):
        self.grid.deposit_pheromone(1, 1, PheromoneType.NEST, 5.0)
        sparse = self.grid.get_sparse_pheromones()
        self.assertTrue(len(sparse) >= 1)
        found = [p for p in sparse if p["x"] == 1 and p["y"] == 1]
        self.assertEqual(len(found), 1)


class TestAntAgent(unittest.TestCase):
    def test_memory_ring_buffer(self):
        agent = AntAgent(id=0, energy=100, memory_size=3)
        for i in range(5):
            agent.push_memory(Position(i, i))
        self.assertEqual(len(agent.memory), 3)
        self.assertEqual(agent.memory[0], Position(2, 2))


class TestValidation(unittest.TestCase):
    def test_valid_config(self):
        cfg = {
            "grid": {"width": 20, "height": 20, "neighborhood": 8, "default_capacity": 99},
            "nest": {"x": 10, "y": 10},
            "food_sources": [{"x": 3, "y": 3, "amount": 50}],
            "agents": {"count": 10, "initial_energy": 200},
            "simulation": {"max_ticks": 100},
        }
        errors = validate_config(cfg)
        self.assertEqual(len(errors), 0)

    def test_missing_keys(self):
        errors = validate_config({})
        self.assertGreater(len(errors), 0)

    def test_invalid_grid(self):
        cfg = {
            "grid": {"width": 1, "height": 1},
            "nest": {"x": 0, "y": 0},
            "food_sources": [],
            "agents": {"count": 1, "initial_energy": 200},
            "simulation": {"max_ticks": 10},
        }
        errors = validate_config(cfg)
        self.assertTrue(any("3x3" in e for e in errors))

    def test_nest_on_obstacle(self):
        cfg = {
            "grid": {"width": 10, "height": 10, "obstacles": [[5, 5]]},
            "nest": {"x": 5, "y": 5},
            "food_sources": [],
            "agents": {"count": 1, "initial_energy": 200},
            "simulation": {"max_ticks": 10},
        }
        errors = validate_config(cfg)
        self.assertTrue(any("obstacle" in e.lower() for e in errors))


class TestAgentLoading(unittest.TestCase):
    def test_load_valid_agent(self):
        code = '''
class TestAgent(AntAgent):
    def decide(self, perception):
        return MoveAction(perception.neighbors[0].x, perception.neighbors[0].y) if perception.neighbors else None
'''
        cls, errors = load_agent_class(code)
        self.assertEqual(len(errors), 0)
        self.assertIsNotNone(cls)

    def test_syntax_error(self):
        cls, errors = load_agent_class("def broken(")
        self.assertIsNone(cls)
        self.assertGreater(len(errors), 0)


class TestBatchExpansion(unittest.TestCase):
    def test_expand(self):
        batch = {
            "parameter": "agents.count",
            "range": [1, 5],
            "step": 2,
            "base_config": {
                "name": "test",
                "agents": {"count": 1},
            },
        }
        configs = expand_batch(batch)
        self.assertEqual(len(configs), 3)  # 1, 3, 5
        self.assertEqual(configs[0]["agents"]["count"], 1)
        self.assertEqual(configs[1]["agents"]["count"], 3)
        self.assertEqual(configs[2]["agents"]["count"], 5)


class TestEngine(unittest.TestCase):
    def _make_config(self, **overrides):
        cfg = {
            "name": "test",
            "grid": {"width": 10, "height": 10, "neighborhood": 8, "default_capacity": 99, "obstacles": []},
            "nest": {"x": 5, "y": 5},
            "food_sources": [{"x": 2, "y": 2, "amount": 5}],
            "agents": {"count": 2, "initial_energy": 50, "memory_size": 3},
            "pheromones": {"drop_strength": 10, "min_drop_floor": 1, "evaporation_rate": 0.1,
                          "evaporation_mode": "exponential", "round_off_threshold": 0.1},
            "simulation": {"max_ticks": 10, "random_seed": 42, "termination": ["max_ticks"],
                          "snapshot_interval": 5, "agent_timeout_ms": 100},
        }
        for k, v in overrides.items():
            if isinstance(v, dict) and k in cfg:
                cfg[k].update(v)
            else:
                cfg[k] = v
        return cfg

    def test_engine_runs(self):
        # Use default agent that returns None (energy drain)
        cfg = self._make_config()
        engine = SimulationEngine(cfg)
        ticks = list(engine.run())
        self.assertGreater(len(ticks), 1)
        self.assertEqual(ticks[0].tick, 0)

    def test_metrics(self):
        cfg = self._make_config()
        engine = SimulationEngine(cfg)
        list(engine.run())
        m = engine.get_metrics()
        self.assertEqual(m.total_ticks, 10)
        self.assertTrue(m.completed)


class TestScoring(unittest.TestCase):
    def test_batch_scores(self):
        metrics = [
            {"food_collected_total": 10, "avg_steps_to_food": 20, "death_ratio": 0.1, "total_ants": 5},
            {"food_collected_total": 20, "avg_steps_to_food": 15, "death_ratio": 0.2, "total_ants": 10},
            {"food_collected_total": 5, "avg_steps_to_food": 30, "death_ratio": 0.0, "total_ants": 3},
        ]
        scores = compute_batch_scores(metrics)
        self.assertEqual(len(scores), 3)
        # Best food collector should have highest norm_food
        self.assertEqual(scores[1].norm_food, 1.0)
        # All percentiles assigned
        percs = sorted([s.percentile for s in scores])
        self.assertAlmostEqual(percs[-1], 1.0)


if __name__ == "__main__":
    unittest.main()
