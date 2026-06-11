"""Metrics — scoring, normalization for batch experiments."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExperimentScore:
    colony_size: int   # X
    norm_food: float
    norm_pathfinding: float
    norm_survival: float
    total_score: float # Y
    percentile: float

    def to_dict(self) -> dict:
        return {
            "colony_size": self.colony_size,
            "norm_food": round(self.norm_food, 4),
            "norm_pathfinding": round(self.norm_pathfinding, 4),
            "norm_survival": round(self.norm_survival, 4),
            "total_score": round(self.total_score, 4),
            "percentile": round(self.percentile, 4),
        }


def compute_batch_scores(all_metrics: list[dict]) -> list[ExperimentScore]:
    """Normalize metrics across a batch and compute composite scores."""
    if not all_metrics:
        return []

    foods = [m.get("food_collected_total", 0) for m in all_metrics]
    stfs = [m.get("avg_steps_to_food", 0) for m in all_metrics]
    deaths = [m.get("death_ratio", 0) for m in all_metrics]

    max_food = max(foods) if foods else 1
    max_stf = max(stfs) if stfs else 1
    max_death = max(deaths) if deaths else 1

    scores = []
    for i, m in enumerate(all_metrics):
        nf = m.get("food_collected_total", 0) / max(1, max_food)
        np_ = 1 - (m.get("avg_steps_to_food", 0) / max(1, max_stf))
        ns = 1 - (m.get("death_ratio", 0) / max(1, max_death))
        total = (nf + np_ + ns) / 3

        scores.append(ExperimentScore(
            colony_size=m.get("total_ants", 0), # X
            norm_food=nf,
            norm_pathfinding=np_,
            norm_survival=ns,
            total_score=total, # Y
            percentile=0,
        ))

    # Compute percentile (rank-based)
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i].total_score)
    for rank, idx in enumerate(sorted_indices):
        scores[idx].percentile = (rank + 1) / len(scores)

    return scores
