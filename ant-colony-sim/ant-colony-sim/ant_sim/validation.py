"""Config + agent .py validation."""
from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

from .models import AntAgent

REQUIRED_KEYS = {"grid", "nest", "food_sources", "agents", "simulation"}


def validate_config(config: dict) -> list[str]:
    """Return list of error strings. Empty = valid."""
    errors: list[str] = []

    for key in REQUIRED_KEYS:
        if key not in config:
            errors.append(f"Missing required key: {key}")

    grid = config.get("grid", {})
    w = grid.get("width", 0)
    h = grid.get("height", 0)
    if w < 3 or h < 3:
        errors.append("Grid must be at least 3x3")
    if w > 200 or h > 200:
        errors.append("Grid max 200x200")

    neighborhood = grid.get("neighborhood", 8)
    if neighborhood not in (4, 8):
        errors.append("neighborhood must be 4 or 8")

    nest = config.get("nest", {})
    nx, ny = nest.get("x", -1), nest.get("y", -1)
    if not (0 <= nx < w and 0 <= ny < h):
        errors.append(f"Nest position ({nx},{ny}) out of grid bounds")

    # Check nest not on obstacle
    obstacles = set()
    for obs in grid.get("obstacles", []):
        if len(obs) >= 2:
            obstacles.add((obs[0], obs[1]))
    if (nx, ny) in obstacles:
        errors.append("Nest cannot be on an obstacle")

    for i, fs in enumerate(config.get("food_sources", [])):
        fx, fy = fs.get("x", -1), fs.get("y", -1)
        if not (0 <= fx < w and 0 <= fy < h):
            errors.append(f"Food source {i} position out of bounds")
        if (fx, fy) in obstacles:
            errors.append(f"Food source {i} on obstacle")
        if fs.get("amount", 0) <= 0:
            errors.append(f"Food source {i} amount must be > 0")

    agents = config.get("agents", {})
    count = agents.get("count", 0)
    if count < 1 or count > 2000:
        errors.append("Agent count must be 1-2000")
    if agents.get("initial_energy", 0) < 10:
        errors.append("initial_energy must be >= 10")

    sim = config.get("simulation", {})
    if sim.get("max_ticks", 0) < 1:
        errors.append("max_ticks must be >= 1")

    phero = config.get("pheromones", {})
    if phero:
        er = phero.get("evaporation_rate", 0.1)
        if not (0 < er <= 1):
            errors.append("evaporation_rate must be in (0, 1]")

    return errors


def load_agent_class(source_code: str) -> tuple[type | None, list[str]]:
    """Load agent class from source code string. Returns (class, errors)."""
    errors: list[str] = []

    # Basic AST check
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        return None, [f"Syntax error: {e}"]

    # Find class that has decide method
    class_name = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name == "decide":
                        class_name = node.name
                        break

    if not class_name:
        return None, ["No class with decide() method found"]

    # Load module
    try:
        mod = types.ModuleType("user_agent")
        mod.__dict__["__builtins__"] = __builtins__

        # Inject ant_sim models into the module
        from . import models
        for name in dir(models):
            if not name.startswith("_"):
                mod.__dict__[name] = getattr(models, name)

        # Also inject common libs
        import math
        import random
        mod.__dict__["math"] = math
        mod.__dict__["random"] = random

        exec(compile(tree, "<agent>", "exec"), mod.__dict__)
        cls = mod.__dict__.get(class_name)
        if cls is None:
            return None, [f"Class {class_name} not found after exec"]

        return cls, []

    except Exception as e:
        return None, [f"Load error: {e}"]


def expand_batch(batch_config: dict) -> list[dict]:
    """Expand batch range syntax to list of configs."""
    param = batch_config["parameter"]
    start, end = batch_config["range"]
    step = batch_config.get("step", 1)
    base = batch_config["base_config"]

    configs = []
    val = start
    while val <= end:
        cfg = _deep_copy_dict(base)
        _set_nested(cfg, param, val)
        cfg["name"] = f"{cfg.get('name', 'batch')}_{param}_{val}"
        configs.append(cfg)
        val += step

    return configs


def _deep_copy_dict(d: dict) -> dict:
    import copy
    return copy.deepcopy(d)


def _set_nested(d: dict, path: str, value: Any) -> None:
    keys = path.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value
