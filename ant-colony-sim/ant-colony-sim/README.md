# Ant Colony Simulation

Agentenbasierte Ameisen-Kolonie-Simulation mit FastAPI Backend, Canvas-Viewer und Matplotlib-Analyse.
Ameisen suchen Futter, folgen Pheromon-Gradienten, und bringen es zum Nest zurück.

## Setup

```bash
# Python 3.10+ erforderlich
pip install -r requirements.txt

# Starten (ohne Auth)
python main.py
# oder
uvicorn main:app --host 0.0.0.0 --port 8000

# Starten (mit API-Key Schutz)
python main.py --api-key dein-geheimer-schluessel
```

Öffne `http://localhost:8000` im Browser.

## Drei Panels

### Panel 1 — Config & Queue

JSON-Editor für die Simulation. Rechts eine Grid-Preview die live aus dem JSON rendert.

**Workflow:**
1. Config anpassen oder `[ default ]` laden
2. Optional: eigenen Agent als `.py` hochladen (Drag & Drop)
3. `[ run ]` — Simulation startet, wechselt automatisch zum Viewer
4. `[ batch ]` — öffnet Modal für Parameter-Sweeps (z.B. `agents.count` von 1 bis 39)

**Config-Felder:**

| Feld | Bedeutung |
|------|-----------|
| `grid.width/height` | Gridgröße (3–200) |
| `grid.obstacles` | Liste von `[x,y]` Hindernissen |
| `nest` | Nest-Position `{x, y}` |
| `food_sources` | Futterquellen `[{x, y, amount}]` |
| `agents.count` | Anzahl Ameisen (1–200) |
| `agents.initial_energy` | Startenergie pro Ameise |
| `agents.memory_size` | Ringpuffer-Größe für besuchte Zellen |
| `pheromones.drop_strength` | Pheromon-Stärke bei Ablage |
| `pheromones.evaporation_rate` | Verdunstungsrate pro Tick (0–1) |
| `simulation.max_ticks` | Maximale Simulationsdauer |
| `simulation.random_seed` | Seed für Reproduzierbarkeit |
| `warmstart.enabled` | Pheromon-Trails beim Start setzen |

### Panel 2 — Viewer & Library

Links: Library mit allen Simulationen (Klick zum Laden).
Mitte: Canvas mit Grid, Ameisen, Pheromonen, Futter.

**Bedienung:**
- `▶ play` / `⏸ pause` — Playback starten/stoppen
- **Scrubber** — zu beliebigem Tick springen
- **Speed-Slider** — 0.00x (Pause) bis 5.00x, Schritte 0.01
- **Klick auf Zelle** — öffnet Tile-Info Panel rechts (Agenten, Pheromone, Food)
- **Klick auf Ameise** — Follow-Mode (blauer Rahmen um tracked Agent)
- **Drag am Rand** — Tile-Info Breite anpassen
- `[ inspect ]` — Tile-Info Panel ein/ausblenden
- `[ screenshot ]` — aktuellen Frame als PNG speichern

**Legende:**
- Weiße Silhouetten = Ameisen
- Gelbe Silhouetten = Ameisen mit Futter
- Orange Heatmap = FOOD Pheromon
- Blaue Heatmap = NEST Pheromon
- Amber Kreise = Futterquellen (Größe = Menge)
- Gestreifte Zellen = Hindernisse
- `N` = Nest

### Panel 3 — Analysis & Export

**Single-Sim Plots:** 1 Sim auswählen (Checkbox), dann:
- `food / time` — Futter gesammelt über Zeit
- `alive / time` — Lebende Ameisen über Zeit
- `steps-to-food` — Histogramm der Schritte bis Futter
- `batch scores` — Composite Score (braucht ≥2 Sims aus Batch)

**Experiment-Vergleich:** 2+ Sims auswählen, dann `food` / `alive` / `steps` — überlagert die Kurven in einem Graph.

**Custom Formula:** Python/Matplotlib Code direkt ausführen.
- `? docs` — öffnet Scope-Referenz mit allen Variablen und klickbaren Beispielen
- **Building Blocks** — Checkboxen für häufige Plots, `[ build ]` generiert den Code
- Syntax-Highlighting im Editor
- Fehler werden direkt auf dem Plot angezeigt (kein Server-Crash)

**Scope-Variablen:** `ax`, `fig`, `plt`, `np`, `metrics`, `ticks`, `config`, `snaps`

**Export:**
- `[ export PNG ]` / `[ export SVG ]` — aktuellen Plot exportieren
- `[ download .sim.zip ]` — Simulation als ZIP (Config + Chunks + Metrics)
- Upload von `.sim.zip` Dateien

## Eigenen Agent schreiben

Erstelle eine `.py` Datei mit einer Klasse die `AntAgent` erbt und `decide()` implementiert:

```python
from ant_sim.models import *
import math, random

class MyAgent(AntAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.my_state = {}  # beliebiger State, lebt die ganze Sim

    def decide(self, perception: Perception) -> Action:
        # perception enthält:
        #   .neighbors      — Liste[NeighborInfo] mit Pheromon/Food/Accessibility
        #   .current_x/y    — aktuelle Position
        #   .carries         — trägt Futter?
        #   .energy          — verbleibende Energie
        #   .current_has_food — Futter auf aktueller Zelle?
        #   .current_is_nest  — auf dem Nest?
        #   .tick            — aktueller Tick
        #
        # self.memory       — list[Position], automatischer Ringpuffer
        #
        # Return EINE Action:
        #   MoveAction(target_x, target_y)    — zu Nachbarzelle bewegen
        #   PickUpAction(source_x, source_y)  — Futter aufheben
        #   DropAction(target_x, target_y)    — Futter am Nest ablegen

        accessible = [n for n in perception.neighbors if n.is_accessible]
        if perception.current_has_food and not perception.carries:
            return PickUpAction(perception.current_x, perception.current_y)
        if perception.current_is_nest and perception.carries:
            return DropAction(perception.current_x, perception.current_y)
        choice = random.choice(accessible)
        return MoveAction(choice.x, choice.y)
```

Hochladen via Drag & Drop auf das Agent-Feld in Panel 1. Ohne Upload wird der `PheromoneFollowerAgent` verwendet.

## Engine-Logik (pro Tick)

1. Pheromone evaporieren
2. Für jede lebende Ameise:
   - Perception bauen → `agent.decide()` aufrufen → Action ausführen
   - Engine deposited automatisch Pheromone (FOOD wenn carrying, sonst NEST)
   - Stärke: `max(min_drop_floor, drop_strength - steps_since_source)`
3. Energie-Refill auf Nest
4. Tod bei Energy ≤ 0
5. Terminierung: `max_ticks` / `no_food_present` / `all_agents_dead`

## API-Key

```bash
python main.py --api-key mein-key
```

Alle `/api/*` Endpoints erfordern dann `Authorization: Bearer mein-key` Header oder `?api_key=mein-key` Query-Param. Das Frontend zeigt automatisch ein Login-Modal.

## Tests

```bash
python -m unittest tests/test_core.py -v
```

## Projektstruktur

```
ant-colony-sim/
├── main.py                  # Uvicorn Entrypoint + --api-key
├── requirements.txt
├── ant_sim/
│   ├── models.py            # Dataclasses, Enums, TickState, Metrics
│   ├── engine.py            # SimulationEngine (Generator pro Tick)
│   ├── grid.py              # GridWorld, Cell, Pheromone-Grid
│   ├── validation.py        # Config + Agent Validierung
│   ├── tick_store.py        # RAM-Window + gzip Disk-Chunks
│   ├── queue.py             # SimQueue (max 4 concurrent)
│   ├── metrics.py           # Batch-Scoring, Normalisierung
│   ├── analysis.py          # Matplotlib Plots + Custom Formula
│   └── api.py               # FastAPI Routes + Auth-Dependency
├── agents/
│   └── default_agent.py     # PheromoneFollowerAgent
├── static/
│   └── index.html           # Single-Page Frontend (TBJS Dark Terminal)
├── data/simulations/        # Gespeicherte Sim-Daten (auto)
└── tests/
    └── test_core.py         # 23 Unit-Tests
```
