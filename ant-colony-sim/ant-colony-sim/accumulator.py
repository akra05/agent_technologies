"""
accumulator.py
--------------
Speichert akkumulierte Statistiken nach jedem Run in acc_state.json.
So kannst du die Simulation beliebig oft neu starten — die Daten
werden einfach weiter aufaddiert, ohne dass alte Runs überschrieben werden.

Workflow:
    1. Simulation läuft → erzeugt Snapshots in einem Ordner
    2. Du rufst:  acc = Accumulator.load()
                  acc.add_run(snapshots)
                  acc.save()
    3. Nächster Run: gleich wieder load() → add_run() → save()
    4. Am Ende:   acc.plot()

Running-Average-Formel:
    neuer_μ = alter_μ + (neuer_wert − alter_μ) / n
    neuer_σ² (Welford) = alter_σ² + (x − alter_μ)(x − neuer_μ)
"""

import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional


# ── Pfade ─────────────────────────────────────────────────────────────────────
DATA_DIR   = pathlib.Path(r"E:\UniSync\6.Semester\Agententechnologie\ant-colony-sim\data\simulations")
STATE_FILE = DATA_DIR.parent / "acc_state.json"   # hier wird akkumuliert
PLOT_FILE  = DATA_DIR.parent / "analysis.png"


# ── Metrik-Helfer (identisch zu analyze_runs.py) ──────────────────────────────

def food_per_step_per_ant(snapshots: list[dict]) -> list[float]:
    result, prev_food = [], 0
    for snap in snapshots:
        current_food = snap.get("nest_food_total", 0)
        alive = max(1, snap.get("alive_count", 1))
        result.append(max(0, current_food - prev_food) / alive)
        prev_food = current_food
    return result


def steps_to_food_per_trip(snapshots: list[dict]) -> list[int]:
    last_empty: dict[int, int] = {}
    trip_lengths: list[int] = []
    for snap in snapshots:
        tick = snap.get("tick", 0)
        for agent in snap.get("agents", []):
            aid = agent["id"]
            carries = agent.get("carries", False)
            if not agent.get("alive", True):
                last_empty.pop(aid, None)
                continue
            if not carries:
                last_empty[aid] = tick
            elif aid in last_empty:
                steps = tick - last_empty[aid]
                if steps > 0:
                    trip_lengths.append(steps)
                del last_empty[aid]
    return trip_lengths


def survival_over_time(snapshots: list[dict]) -> tuple[list[int], list[float]]:
    ticks, rates = [], []
    for snap in snapshots:
        alive = snap.get("alive_count", 0)
        dead  = snap.get("dead_count", 0)
        total = alive + dead
        ticks.append(snap.get("tick", 0))
        rates.append(alive / total if total > 0 else 1.0)
    return ticks, rates


# ── Welford Online-Algorithmus ────────────────────────────────────────────────

class WelfordArray:
    """
    Berechnet Mittelwert + Varianz inkrementell für Arrays gleicher Länge.
    Kein Unterschied ob du alle Daten auf einmal oder einen nach dem anderen schickst.
    """
    def __init__(self, n: int = 0, mean: Optional[list] = None, M2: Optional[list] = None):
        self.n    = n
        self.mean = np.array(mean, dtype=float) if mean else None
        self.M2   = np.array(M2,   dtype=float) if M2   else None

    def update(self, new_values: list[float]):
        x = np.array(new_values, dtype=float)
        if self.mean is None:
            self.mean = np.zeros_like(x)
            self.M2   = np.zeros_like(x)
        # Längen angleichen (kürzerer gewinnt)
        min_len = min(len(x), len(self.mean))
        x         = x[:min_len]
        self.mean = self.mean[:min_len]
        self.M2   = self.M2[:min_len]

        self.n += 1
        delta      = x - self.mean
        self.mean += delta / self.n
        delta2     = x - self.mean
        self.M2   += delta * delta2

    @property
    def std(self) -> np.ndarray:
        if self.n < 2:
            return np.zeros_like(self.mean)
        return np.sqrt(self.M2 / (self.n - 1))

    def to_dict(self) -> dict:
        return {
            "n":    self.n,
            "mean": self.mean.tolist() if self.mean is not None else [],
            "M2":   self.M2.tolist()   if self.M2   is not None else [],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WelfordArray":
        return cls(n=d["n"], mean=d["mean"], M2=d["M2"])


# ── Accumulator ───────────────────────────────────────────────────────────────

class Accumulator:
    """
    Hält den laufenden Durchschnitt über beliebig viele Runs.
    Speichert alles in STATE_FILE — kann also jederzeit unterbrochen werden.
    """

    def __init__(self):
        self.n_runs      = 0
        self.fps_wf      = WelfordArray()   # food per step per ant
        self.surv_wf     = WelfordArray()   # survival rate
        self.ticks       = []

        # Trips: wir speichern Summe + Anzahl für den Mittelwert,
        # und eine komprimierte Verteilung (Bins) für das Histogramm
        self.trip_sum    = 0.0
        self.trip_count  = 0
        self.trip_bins   = np.zeros(200, dtype=int)   # Bin-Breite = 1 Schritt
        self.trip_bin_max = 200                        # alles über 200 → letzter Bin

    # ── Laden / Speichern ──────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: pathlib.Path = STATE_FILE) -> "Accumulator":
        acc = cls()
        if not path.exists():
            print("  Kein gespeicherter Zustand gefunden — starte frisch.")
            return acc
        data = json.loads(path.read_text(encoding="utf-8"))
        acc.n_runs     = data["n_runs"]
        acc.fps_wf     = WelfordArray.from_dict(data["fps"])
        acc.surv_wf    = WelfordArray.from_dict(data["surv"])
        acc.ticks      = data["ticks"]
        acc.trip_sum   = data["trip_sum"]
        acc.trip_count = data["trip_count"]
        acc.trip_bins  = np.array(data["trip_bins"], dtype=int)
        acc.trip_bin_max = data.get("trip_bin_max", 200)
        print(f"  Zustand geladen: {acc.n_runs} Runs akkumuliert.")
        return acc

    def save(self, path: pathlib.Path = STATE_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "n_runs":      self.n_runs,
            "fps":         self.fps_wf.to_dict(),
            "surv":        self.surv_wf.to_dict(),
            "ticks":       self.ticks,
            "trip_sum":    self.trip_sum,
            "trip_count":  self.trip_count,
            "trip_bins":   self.trip_bins.tolist(),
            "trip_bin_max": self.trip_bin_max,
        }
        path.write_text(json.dumps(data), encoding="utf-8")

    # ── Einen Run hinzufügen ───────────────────────────────────────────────────

    def add_run(self, snapshots: list[dict]):
        fps   = food_per_step_per_ant(snapshots)
        trips = steps_to_food_per_trip(snapshots)
        ticks, surv = survival_over_time(snapshots)

        self.fps_wf.update(fps)
        self.surv_wf.update(surv)

        if not self.ticks:
            self.ticks = ticks

        for t in trips:
            self.trip_sum   += t
            self.trip_count += 1
            bin_idx = min(int(t), self.trip_bin_max - 1)
            self.trip_bins[bin_idx] += 1

        self.n_runs += 1
        print(f"  Run {self.n_runs} akkumuliert. Trips bisher: {self.trip_count}")

    # ── Snapshot-Ordner automatisch laden ─────────────────────────────────────

    def add_run_from_folder(self, sim_folder: pathlib.Path):
        snap_dir = sim_folder / "snapshots"
        if not snap_dir.exists():
            print(f"  WARNUNG: Kein snapshots/-Ordner in {sim_folder.name}")
            return
        files = sorted(snap_dir.glob("snapshot_*.json"),
                       key=lambda f: int(f.stem.split("_")[1]))
        snaps = [json.loads(f.read_text(encoding="utf-8")) for f in files]
        self.add_run(snaps)

    # ── Alle Ordner in DATA_DIR einlesen ──────────────────────────────────────
    # Nützlich wenn du schon viele Runs lokal hast und einmalig alles reinladen willst.

    def add_all_from_dir(self, data_dir: pathlib.Path = DATA_DIR):
        folders = sorted(data_dir.iterdir())
        for folder in folders:
            if folder.is_dir():
                self.add_run_from_folder(folder)

    # ── Plot ───────────────────────────────────────────────────────────────────

    def plot(self, out_path: pathlib.Path = PLOT_FILE):
        DARK  = "#000000"; PANEL = "#0a0a0f"; GRID  = "#ffffff0d"
        SPINE = "#ffffff1f"; LABEL = "#808080"; TITLE = "#ebebeb"
        BLUE  = "#4d7fff"; GREEN = "#2dd4a0"; MONO  = "monospace"

        def style_ax(ax):
            ax.set_facecolor(PANEL)
            ax.tick_params(colors=LABEL)
            ax.grid(True, color=GRID, linewidth=0.5)
            for spine in ax.spines.values():
                spine.set_color(SPINE)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig = plt.figure(figsize=(14, 10))
        fig.patch.set_facecolor(DARK)
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
        n  = self.n_runs

        # 1) Essen pro Schritt pro Ameise
        ax1 = fig.add_subplot(gs[0, 0])
        style_ax(ax1)
        ticks = self.ticks[:len(self.fps_wf.mean)]
        ax1.plot(ticks, self.fps_wf.mean, color=BLUE, linewidth=1.5, label=f"Ø (n={n})")
        ax1.fill_between(ticks,
                         self.fps_wf.mean - self.fps_wf.std,
                         self.fps_wf.mean + self.fps_wf.std,
                         alpha=0.2, color=BLUE)
        ax1.set_title("Essen / Schritt / Ameise", color=TITLE, fontfamily=MONO, fontsize=11)
        ax1.set_xlabel("Tick", color=LABEL, fontfamily=MONO, fontsize=9)
        ax1.set_ylabel("Essen pro Schritt", color=LABEL, fontfamily=MONO, fontsize=9)
        ax1.legend(fontsize=8, facecolor=PANEL, edgecolor=SPINE, labelcolor="#cccccc")

        # 2) Histogramm aus Bins
        ax2 = fig.add_subplot(gs[0, 1])
        style_ax(ax2)
        if self.trip_count > 0:
            avg_trip = self.trip_sum / self.trip_count
            xs = np.arange(self.trip_bin_max)
            ax2.bar(xs, self.trip_bins, color=BLUE, alpha=0.85, width=1.0)
            ax2.axvline(avg_trip, color=GREEN, linewidth=1.5, linestyle="--",
                        label=f"Ø {avg_trip:.1f} Schritte")
            ax2.legend(fontsize=8, facecolor=PANEL, edgecolor=SPINE, labelcolor="#cccccc")
        ax2.set_title("Schritte bis zum Essen (alle Trips)", color=TITLE, fontfamily=MONO, fontsize=11)
        ax2.set_xlabel("Schritte", color=LABEL, fontfamily=MONO, fontsize=9)
        ax2.set_ylabel("Anzahl Trips", color=LABEL, fontfamily=MONO, fontsize=9)

        # 3) Überlebensrate
        ax3 = fig.add_subplot(gs[1, 0])
        style_ax(ax3)
        ticks_s = self.ticks[:len(self.surv_wf.mean)]
        ax3.plot(ticks_s, self.surv_wf.mean, color=GREEN, linewidth=1.5, label=f"Ø (n={n})")
        ax3.fill_between(ticks_s,
                         self.surv_wf.mean - self.surv_wf.std,
                         self.surv_wf.mean + self.surv_wf.std,
                         alpha=0.2, color=GREEN)
        ax3.set_title("Überlebensrate", color=TITLE, fontfamily=MONO, fontsize=11)
        ax3.set_xlabel("Tick", color=LABEL, fontfamily=MONO, fontsize=9)
        ax3.set_ylabel("Alive / Total", color=LABEL, fontfamily=MONO, fontsize=9)
        ax3.set_ylim(0, 1.05)
        ax3.legend(fontsize=8, facecolor=PANEL, edgecolor=SPINE, labelcolor="#cccccc")

        # 4) Zusammenfassung
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_facecolor(PANEL)
        ax4.set_xticks([]); ax4.set_yticks([])
        for spine in ax4.spines.values():
            spine.set_color(SPINE)
        avg_trip = self.trip_sum / self.trip_count if self.trip_count > 0 else 0
        summary = (
            f"Runs akkumuliert:      {n}\n"
            f"Trips gesamt:          {self.trip_count}\n"
            f"Ø Schritte bis Essen:  {avg_trip:.1f}\n"
            f"Ø Überlebensrate:      {self.surv_wf.mean.mean():.1%}\n"
            f"Ø Essen/Schritt/Ameis: {self.fps_wf.mean.mean():.4f}\n"
        )
        ax4.text(0.1, 0.5, summary, transform=ax4.transAxes,
                 va="center", color="#cccccc", fontfamily=MONO, fontsize=11, linespacing=2.0)
        ax4.set_title("Zusammenfassung", color=TITLE, fontfamily=MONO, fontsize=11)

        fig.suptitle(f"Ant Colony — Akkumulierte Analyse ({n} Runs)",
                     color=TITLE, fontfamily=MONO, fontsize=13, y=0.98)
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK)
        print(f"\n→ Plot gespeichert: {out_path.resolve()}")
        plt.show()

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset(self, path: pathlib.Path = STATE_FILE):
        """Löscht den gespeicherten Zustand und startet frisch."""
        if path.exists():
            path.unlink()
        self.__init__()
        print("  Zustand zurückgesetzt.")


# ── Standalone-Nutzung ────────────────────────────────────────────────────────

if __name__ == "__main__":
    acc = Accumulator.load()

    # Option A: Alle bereits vorhandenen Ordner einlesen (Einmal-Import)
    if acc.n_runs == 0 and DATA_DIR.exists():
        print(f"\nLese alle Runs aus {DATA_DIR} ein...")
        acc.add_all_from_dir(DATA_DIR)
        acc.save()

    if acc.n_runs == 0:
        print("Keine Runs gefunden. Erst Simulationen durchführen.")
        exit(1)

    print(f"\n{acc.n_runs} Runs akkumuliert — erstelle Plot...")
    acc.plot()
