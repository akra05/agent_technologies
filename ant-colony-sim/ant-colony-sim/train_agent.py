"""Auto-Research Loop — LLM schreibt und verbessert Agents iterativ.

Usage:
    python train_agent.py --iterations 10 --model gpt-4o-mini
    python train_agent.py --iterations 20 --model claude-sonnet-4-20250514 --api-key sk-...
"""
import argparse
import json
import time
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from ant_sim.engine import SimulationEngine
from ant_sim.validation import load_agent_class
from ant_sim.metrics import compute_batch_scores

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Agent Interface Spec for the LLM ──

AGENT_SPEC = """You write Python agent code for an ant colony simulation.

## Interface

Your code must define exactly ONE class inheriting AntAgent with a decide() method.
Available imports (already in scope): AntAgent, Perception, Action, MoveAction,
PickUpAction, DropAction, ActionResult, Position, NeighborInfo, math, random.

```python
class MyAgent(AntAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # custom persistent state here

    def decide(self, perception: Perception) -> Action:
        # MUST return ONE Action per tick
        ...
```

## Perception fields
- perception.neighbors: list[NeighborInfo]  — each has:
    .x, .y, .food_pheromone, .nest_pheromone, .has_food, .food_amount,
    .is_nest, .is_accessible, .agent_count
- perception.current_x, .current_y — agent position
- perception.current_has_food — food on current cell
- perception.current_is_nest — on nest cell
- perception.current_food_pheromone, .current_nest_pheromone
- perception.carries — agent carries food
- perception.energy — remaining energy
- perception.tick — current tick number
- perception.last_action_result — ActionResult (SUCCESS/ILLEGAL/NO_ENERGY) or None

## Agent self fields (persist across ticks, same instance)
- self.memory: list[Position] — auto ring buffer of visited positions (size from config)
- self.x, self.y — current position
- self.energy, self.carries, self.alive
- self.steps_since_source — resets on food pickup or nest visit
- You can add ANY custom attributes in __init__ (they persist)

## Actions (return exactly one)
- MoveAction(target_x, target_y) — must be accessible neighbor
- PickUpAction(source_x, source_y) — current cell must have food, not carrying
- DropAction(target_x, target_y) — current cell must be nest, carrying food

## Engine rules (you cannot control these)
- Engine auto-deposits pheromones: FOOD if carrying, NEST otherwise
- Strength: max(min_drop_floor, drop_strength - steps_since_source)
- Energy: -1 per tick, refill to max on nest
- Death at energy 0

## Scoring (maximize these)
- food_collected: total food brought to nest
- pathfinding: fewer steps to find food = better
- survival: fewer deaths = better
- composite score = mean(norm_food, norm_pathfinding, norm_survival)

## CONSTRAINTS
- Do NOT hardcode nest position (x,y) — find it via NEST pheromones
- Do NOT hardcode food positions — find via FOOD pheromones or exploration
- Agent must work for ANY grid layout, food placement, colony size 2-100
- Keep decide() under 100ms
- Only use stdlib (math, random). No numpy, no external libs.
"""

BASE_CONFIG = {
    "name": "train",
    "grid": {"width": 20, "height": 20, "neighborhood": 8, "default_capacity": 99,
             "obstacles": [[5,3],[5,4],[5,5],[5,6],[6,3],[6,4],[6,5],[6,6],
                           [12,10],[12,11],[12,12],[13,10],[13,11],[13,12]]},
    "nest": {"x": 10, "y": 10},
    "food_sources": [{"x": 3, "y": 3, "amount": 50}, {"x": 17, "y": 15, "amount": 30}],
    "agents": {"count": 10, "initial_energy": 200, "memory_size": 5},
    "pheromones": {"drop_strength": 10, "min_drop_floor": 1, "evaporation_rate": 0.1,
                   "evaporation_mode": "exponential", "round_off_threshold": 0.1},
    "simulation": {"max_ticks": 500, "random_seed": 42,
                   "termination": ["max_ticks", "no_food_present", "all_agents_dead"],
                   "snapshot_interval": 5, "agent_timeout_ms": 100},
}


def run_batch(agent_code: str, sizes: list[int] = None,
              max_ticks: int = 500) -> dict:
    """Run batch of simulations, return aggregated results."""
    if sizes is None:
        sizes = list(range(2, 101, 2))

    cls, errors = load_agent_class(agent_code)
    if errors:
        return {"error": errors, "scores": [], "avg_score": 0.0}

    all_metrics = []
    for n in sizes:
        cfg = json.loads(json.dumps(BASE_CONFIG))
        cfg["agents"]["count"] = n
        cfg["simulation"]["max_ticks"] = max_ticks
        cfg["name"] = f"train_n{n}"

        try:
            engine = SimulationEngine(cfg, cls)
            list(engine.run())
            m = engine.get_metrics().to_dict()
            all_metrics.append(m)
        except Exception as e:
            all_metrics.append({
                "total_ants": n, "food_collected_total": 0,
                "avg_steps_to_food": 999, "death_ratio": 1.0,
            })

    scores = compute_batch_scores(all_metrics)
    score_dicts = [s.to_dict() for s in scores]
    avg = sum(s.total_score for s in scores) / max(1, len(scores))

    return {
        "scores": score_dicts,
        "avg_score": round(avg, 4),
        "best_size": score_dicts[max(range(len(score_dicts)),
                     key=lambda i: score_dicts[i]["total_score"])]["colony_size"] if score_dicts else 0,
        "per_size_summary": [
            {"n": m.get("total_ants", 0),
             "food": m.get("food_collected_total", 0),
             "steps": round(m.get("avg_steps_to_food", 0), 1),
             "death_ratio": round(m.get("death_ratio", 0), 3)}
            for m in all_metrics
        ],
    }


def call_llm(model: str, messages: list[dict], api_key: str = None,
             base_url: str = None) -> str:
    """Call LLM via litellm or raw HTTP."""
    try:
        import litellm
        if api_key:
            litellm.api_key = api_key
        if base_url:
            litellm.api_base = base_url
        resp = litellm.completion(model=model, messages=messages,
                                  max_tokens=4000, temperature=0.7)
        return resp.choices[0].message.content
    except ImportError:
        pass

    # Fallback: raw requests
    import urllib.request
    url = base_url or "https://api.openai.com/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    data = json.dumps({
        "model": model, "messages": messages,
        "max_tokens": 4000, "temperature": 0.7,
    }).encode()

    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    return result["choices"][0]["message"]["content"]


def extract_code(response: str) -> str:
    """Extract Python code from LLM response."""
    # Find ```python ... ``` block
    if "```python" in response:
        start = response.index("```python") + len("```python")
        end = response.index("```", start)
        return response[start:end].strip()
    if "```" in response:
        start = response.index("```") + 3
        # Skip optional language tag
        newline = response.index("\n", start)
        start = newline + 1
        end = response.index("```", start)
        return response[start:end].strip()
    # No code block — return as-is, hope for the best
    return response.strip()


def plot_training(history: list[dict], output: str = "training_curve.png"):
    """Plot training progress."""
    iters = [h["iteration"] for h in history]
    scores = [h["avg_score"] for h in history]
    foods = [h.get("total_food", 0) for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#000000")

    for ax in [ax1, ax2]:
        ax.set_facecolor("#0a0a0f")
        ax.tick_params(colors="#808080")
        ax.spines["bottom"].set_color("#ffffff1f")
        ax.spines["left"].set_color("#ffffff1f")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, color="#ffffff0d", linewidth=0.5)

    # Score curve
    ax1.plot(iters, scores, color="#4d7fff", linewidth=2, marker="o", markersize=4)
    ax1.fill_between(iters, scores, alpha=0.1, color="#4d7fff")
    ax1.set_xlabel("Iteration", color="#808080", fontfamily="monospace")
    ax1.set_ylabel("Avg Composite Score", color="#808080", fontfamily="monospace")
    ax1.set_title("Training Progress", color="#ebebeb", fontfamily="monospace")
    best_i = max(range(len(scores)), key=lambda i: scores[i])
    ax1.annotate(f"best: {scores[best_i]:.3f} (iter {iters[best_i]})",
                  xy=(iters[best_i], scores[best_i]),
                  xytext=(iters[best_i], scores[best_i] + 0.02),
                  color="#2dd4a0", fontfamily="monospace", fontsize=9,
                  arrowprops=dict(arrowstyle="->", color="#2dd4a0"))

    # Food curve
    ax2.plot(iters, foods, color="#ffaa28", linewidth=2, marker="s", markersize=4)
    ax2.fill_between(iters, foods, alpha=0.1, color="#ffaa28")
    ax2.set_xlabel("Iteration", color="#808080", fontfamily="monospace")
    ax2.set_ylabel("Total Food (summed across batch)", color="#808080", fontfamily="monospace")
    ax2.set_title("Food Collection", color="#ebebeb", fontfamily="monospace")

    fig.tight_layout()
    fig.savefig(output, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[plot] saved → {output}")


def train(iterations: int, model: str, api_key: str = None,
          base_url: str = None, batch_sizes: list[int] = None,
          max_ticks: int = 500, output_dir: str = "training_runs"):
    """Main training loop."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if batch_sizes is None:
        batch_sizes = list(range(2, 101, 2))

    history: list[dict] = []
    best_score = -1.0
    best_code = ""
    conversation: list[dict] = [
        {"role": "system", "content": AGENT_SPEC},
    ]

    print(f"[train] model={model}, iterations={iterations}, "
          f"batch={len(batch_sizes)} sizes, ticks={max_ticks}")
    print("=" * 60)

    # ── Iteration 0: initial agent ──
    conversation.append({
        "role": "user",
        "content": (
            "Write an initial ant foraging agent. Focus on:\n"
            "1. Explore efficiently using FOOD pheromone gradients\n"
            "2. Return to nest using NEST pheromone gradients\n"
            "3. Avoid loops using self.memory\n"
            "4. Balance exploration vs exploitation\n\n"
            "Return ONLY a ```python``` code block with the agent class."
        ),
    })

    for iteration in range(iterations):
        t0 = time.time()
        print(f"\n[iter {iteration}] asking LLM...")

        try:
            response = call_llm(model, conversation, api_key, base_url)
            code = extract_code(response)
        except Exception as e:
            print(f"[iter {iteration}] LLM error: {e}")
            continue

        # Save code
        code_path = out / f"agent_iter{iteration:03d}.py"
        code_path.write_text(code, encoding="utf-8")
        print(f"[iter {iteration}] agent saved → {code_path.name}")

        # Run batch
        print(f"[iter {iteration}] running batch ({len(batch_sizes)} sims)...")
        results = run_batch(code, batch_sizes, max_ticks)

        if "error" in results and results["error"]:
            print(f"[iter {iteration}] code error: {results['error']}")
            conversation.append({"role": "assistant", "content": f"```python\n{code}\n```"})
            conversation.append({
                "role": "user",
                "content": (
                    f"The agent code has errors:\n{results['error']}\n\n"
                    "Fix the errors and return the corrected ```python``` code block."
                ),
            })
            history.append({"iteration": iteration, "avg_score": 0.0,
                           "total_food": 0, "status": "error"})
            continue

        avg = results["avg_score"]
        total_food = sum(s["food"] for s in results["per_size_summary"])
        dt = time.time() - t0

        # Track best
        improved = avg > best_score
        if improved:
            best_score = avg
            best_code = code

        history.append({
            "iteration": iteration, "avg_score": avg,
            "total_food": total_food, "best_size": results["best_size"],
            "time_s": round(dt, 1), "status": "improved" if improved else "no_improvement",
        })

        print(f"[iter {iteration}] score={avg:.4f} food={total_food} "
              f"best_n={results['best_size']} {'★ NEW BEST' if improved else ''} "
              f"({dt:.1f}s)")

        # Top/bottom 3 sizes for feedback
        summary = results["per_size_summary"]
        top3 = sorted(summary, key=lambda s: s["food"], reverse=True)[:3]
        bot3 = sorted(summary, key=lambda s: s["food"])[:3]

        # Build feedback for next iteration
        conversation.append({"role": "assistant", "content": f"```python\n{code}\n```"})
        conversation.append({
            "role": "user",
            "content": (
                f"## Results (iteration {iteration})\n"
                f"- Avg composite score: {avg:.4f} (best ever: {best_score:.4f})\n"
                f"- Total food across batch: {total_food}\n"
                f"- Best colony size: n={results['best_size']}\n"
                f"\nTop 3 sizes: {json.dumps(top3)}\n"
                f"Bottom 3 sizes: {json.dumps(bot3)}\n"
                f"\nFull per-size data: {json.dumps(summary[:10])}...\n\n"
                "Analyze what works and what doesn't. Then write an IMPROVED agent.\n"
                "Consider: pheromone following strength, exploration rate, "
                "memory usage, state machine transitions, energy management.\n\n"
                "Return ONLY the improved ```python``` code block."
            ),
        })

        # Keep conversation manageable (last 6 exchanges max)
        if len(conversation) > 13:
            conversation = [conversation[0]] + conversation[-12:]

        # Save progress
        (out / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8")
        plot_training(history, str(out / "training_curve.png"))

    # Save best agent
    if best_code:
        best_path = out / "best_agent.py"
        best_path.write_text(best_code, encoding="utf-8")
        print(f"\n{'=' * 60}")
        print(f"[done] best score: {best_score:.4f}")
        print(f"[done] best agent → {best_path}")
        print(f"[done] training curve → {out / 'training_curve.png'}")
        print(f"[done] all iterations → {out / 'history.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Agent Training Loop")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--model", type=str, default="openrouter/tencent/hy3-preview")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--max-ticks", type=int, default=500)
    parser.add_argument("--min-ants", type=int, default=2)
    parser.add_argument("--max-ants", type=int, default=100)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--output", type=str, default="training_runs")
    args = parser.parse_args()

    sizes = list(range(args.min_ants, args.max_ants + 1, args.step))
    train(
        iterations=args.iterations,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        batch_sizes=sizes,
        max_ticks=args.max_ticks,
        output_dir=args.output,
    )
"""

**Usage:**

```bash
# OpenAI
python train_agent.py --iterations 15 --model gpt-4o-mini --api-key sk-...

# Anthropic via LiteLLM
python train_agent.py --iterations 10 --model claude-sonnet-4-20250514 --api-key sk-ant-...

# Lokales Modell
python train_agent.py --iterations 20 --model local/model --base-url http://localhost:8080/v1

# Kleiner Batch zum Testen
python train_agent.py --iterations 3 --model gpt-4o-mini --min-ants 5 --max-ants 20 --step 5 --max-ticks 200
```

**Output in `training_runs/`:**
- `agent_iter000.py` bis `agent_iterN.py` — jeder Agent-Versuch
- `best_agent.py` — bester Agent über alle Iterationen
- `history.json` — Score/Food/Time pro Iteration
- `training_curve.png` — zwei Plots: Composite Score + Total Food über Iterationen

**Loop-Ablauf:**
1. LLM schreibt initialen Agent
2. Batch-Run (50 Sims: 2,4,6...100 Ameisen)
3. Scores + Top/Bottom 3 Sizes → zurück ans LLM als Feedback
4. LLM analysiert + schreibt verbesserten Agent
5. Repeat. Conversation wird auf letzte 6 Exchanges gekappt (Context-Management)
6. Code-Fehler werden automatisch zurückgefüttert ("fix these errors")"""
