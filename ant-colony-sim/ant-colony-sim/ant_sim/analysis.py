"""AnalysisEngine — formula exec + matplotlib plot generation."""
from __future__ import annotations

import csv
import io
import uuid
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class AnalysisEngine:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.plots_dir = data_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def builtin_food_over_time(self, metrics: dict) -> tuple[str, bytes]:
        """Food collected over time from periodic snapshots."""
        snapshots = metrics.get("snapshots", [])
        ticks = [s["tick"] for s in snapshots]
        food = [s["food_collected_total"] for s in snapshots]

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#000000")
        ax.set_facecolor("#0a0a0f")
        ax.plot(ticks, food, color="#4d7fff", linewidth=1.5)
        ax.set_xlabel("Tick", color="#808080", fontfamily="monospace", fontsize=10)
        ax.set_ylabel("Food Collected", color="#808080", fontfamily="monospace", fontsize=10)
        ax.set_title(f"# Food / Time — {metrics.get('name', '')}", color="#ebebeb",
                      fontfamily="monospace", fontsize=12)
        ax.tick_params(colors="#808080")
        ax.spines["bottom"].set_color("#ffffff1f")
        ax.spines["left"].set_color("#ffffff1f")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, color="#ffffff0d", linewidth=0.5)

        return self._save_plot(fig, "food_time")

    def builtin_alive_over_time(self, metrics: dict) -> tuple[str, bytes]:
        snapshots = metrics.get("snapshots", [])
        ticks = [s["tick"] for s in snapshots]
        alive = [s["alive_count"] for s in snapshots]

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#000000")
        ax.set_facecolor("#0a0a0f")
        ax.plot(ticks, alive, color="#2dd4a0", linewidth=1.5)
        ax.set_xlabel("Tick", color="#808080", fontfamily="monospace", fontsize=10)
        ax.set_ylabel("Alive Agents", color="#808080", fontfamily="monospace", fontsize=10)
        ax.set_title(f"# Alive / Time — {metrics.get('name', '')}", color="#ebebeb",
                      fontfamily="monospace", fontsize=12)
        ax.tick_params(colors="#808080")
        ax.spines["bottom"].set_color("#ffffff1f")
        ax.spines["left"].set_color("#ffffff1f")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, color="#ffffff0d", linewidth=0.5)

        return self._save_plot(fig, "alive_time")

    def builtin_steps_to_food(self, metrics: dict) -> tuple[str, bytes]:
        trips = metrics.get("trips", [])
        if not trips:
            return self._empty_plot("No trips recorded")

        steps = [t["steps_to_food"] for t in trips]

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#000000")
        ax.set_facecolor("#0a0a0f")
        ax.hist(steps, bins=min(30, len(steps)), color="#4d7fff", edgecolor="#0a0a0f", alpha=0.8)
        ax.set_xlabel("Steps to Food", color="#808080", fontfamily="monospace", fontsize=10)
        ax.set_ylabel("Count", color="#808080", fontfamily="monospace", fontsize=10)
        ax.set_title(f"# Steps-to-Food Distribution — {metrics.get('name', '')}", color="#ebebeb",
                      fontfamily="monospace", fontsize=12)
        ax.tick_params(colors="#808080")
        ax.spines["bottom"].set_color("#ffffff1f")
        ax.spines["left"].set_color("#ffffff1f")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        return self._save_plot(fig, "steps_food")

    def builtin_batch_scores(self, scores: list[dict]) -> tuple[str, bytes]:
        if len(scores) < 2:
            return self._empty_plot("Batch scores need >= 2 sims (run a batch first)")

        sizes = [s["colony_size"] for s in scores]
        totals = [s["total_score"] for s in scores]

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#000000")
        ax.set_facecolor("#0a0a0f")
        ax.bar(range(len(sizes)), totals, color="#4d7fff", edgecolor="#0a0a0f", width=0.8)
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels([str(s) for s in sizes], fontsize=8)
        ax.set_xlabel("Colony Size", color="#808080", fontfamily="monospace", fontsize=10)
        ax.set_ylabel("Composite Score (mean of food+pathfinding+survival)",
                       color="#808080", fontfamily="monospace", fontsize=10)
        ax.set_title("# Batch Composite Scores", color="#ebebeb",
                      fontfamily="monospace", fontsize=12)
        ax.tick_params(colors="#808080")
        ax.spines["bottom"].set_color("#ffffff1f")
        ax.spines["left"].set_color("#ffffff1f")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", color="#ffffff0d", linewidth=0.5)

        return self._save_plot(fig, "batch_scores")

    def compare_sims(self, all_metrics: list[dict], metric_key: str = "food") -> tuple[str, bytes]:
        """Overlay multiple sims in one plot for comparison."""
        if not all_metrics:
            return self._empty_plot("No metrics to compare")

        colors = ["#4d7fff", "#2dd4a0", "#ff6b6b", "#ffcc44",
                  "#a78bfa", "#f472b6", "#38bdf8", "#fb923c",
                  "#34d399", "#818cf8"]

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#000000")
        ax.set_facecolor("#0a0a0f")

        key_map = {
            "food": ("food_collected_total", "Food Collected"),
            "alive": ("alive_count", "Alive Agents"),
            "steps": ("avg_steps_to_food", "Avg Steps to Food"),
        }
        snap_key, ylabel = key_map.get(metric_key, ("food_collected_total", "Food Collected"))

        for i, m in enumerate(all_metrics):
            snaps = m.get("snapshots", [])
            if not snaps:
                continue
            ticks = [s["tick"] for s in snaps]
            vals = [s.get(snap_key, 0) for s in snaps]
            label = f"{m.get('name', f'sim_{i}')} (n={m.get('total_ants', '?')})"
            c = colors[i % len(colors)]
            ax.plot(ticks, vals, color=c, linewidth=1.2, label=label, alpha=0.85)

        ax.set_xlabel("Tick", color="#808080", fontfamily="monospace", fontsize=10)
        ax.set_ylabel(ylabel, color="#808080", fontfamily="monospace", fontsize=10)
        ax.set_title(f"# Experiment Comparison — {ylabel}", color="#ebebeb",
                      fontfamily="monospace", fontsize=12)
        ax.tick_params(colors="#808080")
        ax.spines["bottom"].set_color("#ffffff1f")
        ax.spines["left"].set_color("#ffffff1f")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, color="#ffffff0d", linewidth=0.5)
        ax.legend(fontsize=8, facecolor="#0a0a0f", edgecolor="#ffffff1f",
                   labelcolor="#cccccc", loc="best")

        return self._save_plot(fig, "compare")

    def run_custom(self, code: str, contexts: list[dict] | dict) -> tuple[str, bytes]:
        """Execute custom analysis code with matplotlib.

        Variables injected per sim:
          First sim  → metrics, ticks, config, snaps  (also metrics_1 etc.)
          Nth sim    → metrics_N, ticks_N, config_N, snaps_N
          All sims   → all_metrics, all_ticks, all_configs (lists)
        """
        # Normalize: single dict → list
        if isinstance(contexts, dict):
            contexts = [contexts] if contexts else []

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#000000")
        ax.set_facecolor("#0a0a0f")

        local_ctx: dict[str, Any] = {"plt": plt, "np": np, "fig": fig, "ax": ax}

        # Collect lists for convenience
        all_metrics, all_ticks, all_configs = [], [], []

        for i, ctx in enumerate(contexts):
            m = ctx.get("metrics", {})
            t = ctx.get("ticks", [])
            c = ctx.get("config", {})
            s = (m or {}).get("snapshots", [])

            all_metrics.append(m)
            all_ticks.append(t)
            all_configs.append(c)

            suffix = f"_{i + 1}"
            local_ctx[f"metrics{suffix}"] = m
            local_ctx[f"ticks{suffix}"] = t
            local_ctx[f"config{suffix}"] = c
            local_ctx[f"snaps{suffix}"] = s

            # First sim also gets bare names (no suffix)
            if i == 0:
                local_ctx.update(metrics=m, ticks=t, config=c, snaps=s)

        local_ctx["all_metrics"] = all_metrics
        local_ctx["all_ticks"] = all_ticks
        local_ctx["all_configs"] = all_configs
        local_ctx["n_sims"] = len(contexts)

        try:
            exec(code, {"__builtins__": __builtins__}, local_ctx)
        except Exception as e:
            # Render the error on the plot so user sees it
            ax.clear()
            ax.set_facecolor("#0a0a0f")
            ax.text(0.5, 0.5, f"Error: {type(e).__name__}\n{e}",
                     transform=ax.transAxes, ha="center", va="center",
                     color="#ff6b6b", fontsize=12, fontfamily="monospace",
                     wrap=True)
            ax.set_xticks([])
            ax.set_yticks([])

        ax.tick_params(colors="#808080")
        for spine in ax.spines.values():
            spine.set_color("#ffffff1f")

        return self._save_plot(fig, "custom")

    def _save_plot(self, fig: plt.Figure, prefix: str) -> tuple[str, bytes]:
        plot_id = f"{prefix}_{uuid.uuid4().hex[:8]}"

        # PNG
        buf_png = io.BytesIO()
        fig.savefig(buf_png, format="png", dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        png_bytes = buf_png.getvalue()

        # SVG
        buf_svg = io.BytesIO()
        fig.savefig(buf_svg, format="svg", bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        svg_bytes = buf_svg.getvalue()

        # CSV (extracted from axes)
        csv_text = self._extract_csv(fig)

        # Save to disk
        (self.plots_dir / f"{plot_id}.png").write_bytes(png_bytes)
        (self.plots_dir / f"{plot_id}.svg").write_bytes(svg_bytes)
        (self.plots_dir / f"{plot_id}.csv").write_text(csv_text, encoding="utf-8")

        plt.close(fig)
        return plot_id, png_bytes

    @staticmethod
    def _extract_csv(fig: plt.Figure) -> str:
        """Extract plotted data from all axes as long-format CSV.

        Columns: axis, series, x, y. Works for plot/scatter (Line2D) and
        hist/bar (BarContainer). Returns header-only CSV when nothing
        extractable is present (e.g. text-only error plots).
        """
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(["axis", "series", "x", "y"])

        for ax_idx, ax in enumerate(fig.axes):
            # Line2D objects: plot(), scatter() via plot
            for line in ax.get_lines():
                label = line.get_label() or "series"
                if label.startswith("_"):
                    label = f"series_{ax_idx}"
                xs, ys = line.get_xdata(), line.get_ydata()
                for x, y in zip(xs, ys):
                    w.writerow([ax_idx, label, x, y])

            # BarContainer: hist(), bar()
            for c_idx, container in enumerate(getattr(ax, "containers", [])):
                label = getattr(container, "get_label", lambda: "")() or f"bars_{c_idx}"
                if label.startswith("_"):
                    label = f"bars_{c_idx}"
                patches = getattr(container, "patches", None) or list(container)
                for patch in patches:
                    if not hasattr(patch, "get_x"):
                        continue
                    x_center = patch.get_x() + patch.get_width() / 2
                    y_val = patch.get_height()
                    w.writerow([ax_idx, label, x_center, y_val])

        return out.getvalue()

    def _empty_plot(self, msg: str) -> tuple[str, bytes]:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#000000")
        ax.set_facecolor("#0a0a0f")
        ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center",
                 color="#808080", fontsize=14, fontfamily="monospace")
        ax.set_xticks([])
        ax.set_yticks([])
        return self._save_plot(fig, "empty")

    def get_plot_path(self, plot_id: str, fmt: str = "png") -> Path | None:
        p = self.plots_dir / f"{plot_id}.{fmt}"
        return p if p.exists() else None
