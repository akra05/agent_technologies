"""
run_batch.py
------------
Vergleicht mehrere Koloniegrössen mit Rolling Average über:
  - Steps to Food (Nest → Nahrung)
  - Steps to Nest (Nahrung → Nest)
  - Death Ratio
"""

import sys
import pathlib
import shutil
import datetime

# ── Cache leeren BEVOR irgendwas importiert wird ──────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).parent

for pycache in PROJECT_ROOT.rglob("__pycache__"):
    shutil.rmtree(pycache, ignore_errors=True)
    print(f"[cache] gelöscht: {pycache}")

for pyc in PROJECT_ROOT.rglob("*.pyc"):
    pyc.unlink(missing_ok=True)
    print(f"[cache] gelöscht: {pyc}")

# ── Jetzt erst Simulation importieren ─────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(PROJECT_ROOT))
from ant_sim.engine import SimulationEngine

# ── Konfiguration ─────────────────────────────────────────────────────────────
N_RUNS        = 10
ROLLING_TICKS = 50

ANT_COUNTS = [5, 10, 15, 20, 30, 40]

BASE_CONFIG = {
    "name": "batch",
    "grid":   {"width": 20, "height": 20},
    "agents": {"initial_energy": 200},
    "simulation": {
        "max_ticks": 500,
        "snapshot_interval": 5,
    },
    "nest": {"x": 10, "y": 10},
}

COLORS = [plt.cm.plasma(i / max(len(ANT_COUNTS) - 1, 1)) for i in range(len(ANT_COUNTS))]

print(f"\n{'='*60}")
print(f"  ANT_COUNTS = {ANT_COUNTS}")
print(f"  N_RUNS     = {N_RUNS}")
print(f"{'='*60}\n")


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def rolling_avg(trips_by_tick: dict, ticks: list, window: int) -> list:
    result = []
    for t in ticks:
        window_trips = []
        for wt, trips in trips_by_tick.items():
            if t - window <= wt <= t:
                window_trips.extend(trips)
        result.append(np.mean(window_trips) if window_trips else np.nan)
    return result


def death_ratio_curve(snapshots) -> list:
    curve = []
    for s in snapshots:
        alive = getattr(s, "alive_count", 0)
        dead  = getattr(s, "dead_count",  0)
        total = alive + dead
        curve.append(dead / total if total > 0 else np.nan)
    return curve


def accumulate_curves(all_curves):
    min_len = min(len(c[1]) for c in all_curves)
    ticks_cut = all_curves[0][0][:min_len]
    matrix = np.array([c[1][:min_len] for c in all_curves])
    return ticks_cut, np.nanmean(matrix, axis=0), np.nanstd(matrix, axis=0)


# ── Batch ─────────────────────────────────────────────────────────────────────

results_steps  = {}
results_return = {}
results_dratio = {}

for ant_count in ANT_COUNTS:
    print(f"\n── {ant_count} Ameisen ──────────────────────────────")
    all_step_curves   = []
    all_return_curves = []
    all_ratio_curves  = []

    for i in range(N_RUNS):
        cfg = {
            **BASE_CONFIG,
            "agents": {**BASE_CONFIG["agents"], "count": ant_count},
            "simulation": {**BASE_CONFIG["simulation"], "random_seed": i},
            "food_sources": [
                {"x": 3,  "y": 3,  "amount": ant_count * 10},
                {"x": 17, "y": 15, "amount": ant_count * 8},
            ],
        }

        engine = SimulationEngine(cfg, agent_class=None)
        if i == 0:
            print(f"  [DEBUG2] engine.agent_count = {engine.agent_count}")
            print(f"  [DEBUG2] cfg['agents'] = {cfg['agents']}")
        # ── DEBUG: Ameisenzahl verifizieren (nur Run 0) ──────────────────────
        if i == 0:
            actual_count = len(engine.agents) if hasattr(engine, "agents") else "???"
            print(f"  [DEBUG] Konfiguriert: {ant_count}  |  Tatsächlich in Engine: {actual_count}")
            if actual_count != ant_count:
                print(f"  [WARNUNG] Diskrepanz! Engine ignoriert 'count' im Config!")

        for _ in engine.run():
            pass

        # Hinrichtung: Nest → Nahrung
        trips_by_tick:  dict[int, list[int]] = {}
        # Rückrichtung: Nahrung → Nest
        return_by_tick: dict[int, list[int]] = {}

        for trip in engine.trips:
            if trip.steps_to_food > 0:
                trips_by_tick.setdefault(trip.t_find_food, []).append(trip.steps_to_food)
            steps_return = trip.t_return_nest - trip.t_find_food
            if steps_return > 0:
                return_by_tick.setdefault(trip.t_return_nest, []).append(steps_return)

        ticks = [s.tick for s in engine.snapshots]
        all_step_curves.append((ticks,  rolling_avg(trips_by_tick,  ticks, ROLLING_TICKS)))
        all_return_curves.append((ticks, rolling_avg(return_by_tick, ticks, ROLLING_TICKS)))
        all_ratio_curves.append((ticks,  death_ratio_curve(engine.snapshots)))

        if (i + 1) % 10 == 0:
            print(f"  Run {i+1:>3}/{N_RUNS} fertig")

    results_steps[ant_count]  = accumulate_curves(all_step_curves)
    results_return[ant_count] = accumulate_curves(all_return_curves)
    results_dratio[ant_count] = accumulate_curves(all_ratio_curves)


# ── Plot ───────────────────────────────────────────────────────────────────────

DARK  = "#000000"; PANEL = "#1a1a2e"; GRID  = "#ffffff0d"
SPINE = "#ffffff1f"; LABEL = "#808080"; TITLE = "#ebebeb"; MONO = "monospace"

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
fig.patch.set_facecolor(DARK)

def style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=LABEL)
    ax.grid(True, color=GRID, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Subplot 1: Nest → Nahrung
style_ax(ax1)
for (ant_count, (ticks, mean, std)), color in zip(results_steps.items(), COLORS):
    ax1.fill_between(ticks, mean - std, mean + std, alpha=0.35, color=color)
    ax1.plot(ticks, mean, color=color, linewidth=2.0, label=f"{ant_count} Ameisen")
ax1.set_title("Nest → Nahrung  (Steps to Food)", color=TITLE, fontfamily=MONO, fontsize=11)
ax1.set_xlabel("Tick", color=LABEL, fontfamily=MONO, fontsize=10)
ax1.set_ylabel(f"Ø Schritte (Rolling {ROLLING_TICKS} Ticks)", color=LABEL, fontfamily=MONO, fontsize=10)
ax1.legend(fontsize=9, facecolor=PANEL, edgecolor=SPINE, labelcolor="#cccccc")

# Subplot 2: Nahrung → Nest
style_ax(ax2)
for (ant_count, (ticks, mean, std)), color in zip(results_return.items(), COLORS):
    ax2.fill_between(ticks, mean - std, mean + std, alpha=0.15, color=color)
    ax2.plot(ticks, mean, color=color, linewidth=2.0, label=f"{ant_count} Ameisen")
ax2.set_title("Nahrung → Nest  (Steps to Return)", color=TITLE, fontfamily=MONO, fontsize=11)
ax2.set_xlabel("Tick", color=LABEL, fontfamily=MONO, fontsize=10)
ax2.set_ylabel(f"Ø Schritte (Rolling {ROLLING_TICKS} Ticks)", color=LABEL, fontfamily=MONO, fontsize=10)
ax2.legend(fontsize=9, facecolor=PANEL, edgecolor=SPINE, labelcolor="#cccccc")

# Subplot 3: Death Ratio
style_ax(ax3)
for (ant_count, (ticks, mean, std)), color in zip(results_dratio.items(), COLORS):
    ax3.fill_between(ticks, mean - std, mean + std, alpha=0.15, color=color)
    ax3.plot(ticks, mean, color=color, linewidth=2.0, label=f"{ant_count} Ameisen")
ax3.set_title("Death Ratio  (tote / gesamt)", color=TITLE, fontfamily=MONO, fontsize=11)
ax3.set_xlabel("Tick", color=LABEL, fontfamily=MONO, fontsize=10)
ax3.set_ylabel("Anteil Tote (0 = keine, 1 = alle)", color=LABEL, fontfamily=MONO, fontsize=10)
ax3.set_ylim(0, 1)
ax3.legend(fontsize=9, facecolor=PANEL, edgecolor=SPINE, labelcolor="#cccccc")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
fig.suptitle(
    f"{N_RUNS} Runs je Konfiguration  |  ANT_COUNTS={ANT_COUNTS}  |  {timestamp}",
    color=LABEL, fontfamily=MONO, fontsize=10, y=1.01
)
plt.tight_layout()

# Timestamp im Dateinamen → nie versehentlich alte Version anschauen
out = PROJECT_ROOT / f"batch_comparison_{timestamp}.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close(fig)
print(f"\n→ Plot gespeichert: {out}")