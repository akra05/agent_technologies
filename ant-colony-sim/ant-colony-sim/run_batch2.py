"""
run_batch.py
------------
Führt N Simulationen durch und plottet den gemittelten
avg_steps_to_food über die Zeit (mit Konfidenzband).

Ausführen (im Projektroot):
    python run_batch.py
"""

import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from ant_sim.engine import SimulationEngine

# ── Konfiguration ─────────────────────────────────────────────────────────────
N_RUNS = 100

CONFIG = {
    "name": "batch",
    "grid":   {"width": 20, "height": 20},
    "agents": {"count": 60, "initial_energy": 200},
    "simulation": {
        "max_ticks": 500,
        "snapshot_interval": 5,   # alle 5 Ticks ein Datenpunkt
    },
    "food_sources": [
        {"x": 3,  "y": 3,  "amount": 10000},
        {"x": 17, "y": 15, "amount": 10000},
    ],
    "nest": {"x": 10, "y": 10},
}

# ── Batch laufen lassen ───────────────────────────────────────────────────────
# Pro Run sammeln wir die snapshot-Kurve: [(tick, avg_steps_to_food), ...]
all_curves = []   # eine Liste pro Run

for i in range(N_RUNS):
    cfg = {**CONFIG, "simulation": {**CONFIG["simulation"], "random_seed": i}}
    engine = SimulationEngine(cfg, agent_class=None)

    for _ in engine.run():
        pass   # Generator komplett durchlaufen

    # engine.snapshots ist eine Liste von PeriodicSnapshot mit .tick und .avg_steps_to_food
    curve = [(s.tick, s.avg_steps_to_food) for s in engine.snapshots]
    all_curves.append(curve)

    last_avg = curve[-1][1] if curve else 0
    print(f"  Run {i+1:>3}/{N_RUNS}  |  Snapshots: {len(curve)}  |  letzter Ø: {last_avg:.1f}")

# ── Kurven auf gleiche Länge bringen und mitteln ──────────────────────────────
# Kürzeste Kurve bestimmt die Länge (falls ein Run früher endet)
min_len = min(len(c) for c in all_curves)
ticks   = [all_curves[0][t][0] for t in range(min_len)]   # Tick-Werte von Run 0

matrix = np.array([[c[t][1] for t in range(min_len)] for c in all_curves])
# matrix shape: (N_RUNS, min_len)

mean = matrix.mean(axis=0)
std  = matrix.std(axis=0)

# ── Plot ───────────────────────────────────────────────────────────────────────
DARK  = "#000000"; PANEL = "#0a0a0f"; GRID  = "#ffffff0d"
SPINE = "#ffffff1f"; LABEL = "#808080"; TITLE = "#ebebeb"
BLUE  = "#4d7fff"; GREEN = "#2dd4a0"; MONO  = "monospace"

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(DARK)
ax.set_facecolor(PANEL)
ax.tick_params(colors=LABEL)
ax.grid(True, color=GRID, linewidth=0.5)
for spine in ax.spines.values():
    spine.set_color(SPINE)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Einzelne Runs ganz dünn im Hintergrund (optional, auskommentieren wenn zu unübersichtlich)
for curve in all_curves:
    xs = [p[0] for p in curve[:min_len]]
    ys = [p[1] for p in curve[:min_len]]
    ax.plot(xs, ys, color=BLUE, alpha=0.05, linewidth=0.8)

# Gemittelter Verlauf + Konfidenzband (±1 Standardabweichung)
ax.fill_between(ticks, mean - std, mean + std, alpha=0.25, color=BLUE, label="±1 Std")
ax.plot(ticks, mean, color=BLUE, linewidth=2.0, label=f"Ø über {N_RUNS} Runs")

ax.set_title(f"Avg Steps to Food über die Zeit  ({N_RUNS} Runs, je 500 Ticks)",
             color=TITLE, fontfamily=MONO, fontsize=13)
ax.set_xlabel("Tick", color=LABEL, fontfamily=MONO, fontsize=10)
ax.set_ylabel("Avg Steps to Food", color=LABEL, fontfamily=MONO, fontsize=10)
ax.legend(fontsize=9, facecolor=PANEL, edgecolor=SPINE, labelcolor="#cccccc")

out = PROJECT_ROOT / "avg_steps_over_time.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK)
print(f"\n→ Plot gespeichert: {out}")
plt.show()
