"""PheromoneFollowerAgent — default agent with full state machine.

State machine: EXPLORE → PICKUP → RETURN → DROP → loop.
- EXPLORE: follow FOOD pheromone gradient probabilistically, random ε=0.15, avoid memory
- RETURN: follow NEST pheromone gradient, fallback toward nest via memory
- Pheromone deposit: engine-automatic (not agent decision per v3 spec)

Sources:
  Dorigo & Stützle, "Ant Colony Optimization" (MIT Press)
  Czaczkes et al. 2024 — differential pheromone deposition (Lasius niger)
  Frankel et al. 2022 (eLife) — crop-dependent biased random walk
  Stickland 1995 — binary bias model
"""

"""Ant Colony Agent — Adaptive Forager v11 (Polyethism — Verhaltensdiversität)

States: LEAVE_NEST → EXPLORE → CHASE_FOOD → RETURN_HOME → RETURN_TO_FOOD → (loop)
                ↘ STARVE ↗                              ↗ STARVE ↗

Phasen (definieren Jitter, Direction-Weight, Survival-Schwellen, etc.):
  Phase 0 — SCOUT       : low jitter, geradeaus, geduldig (timeout 100)
  Phase 1 — BUILDER     : medium jitter, desperate aktiv (timeout 80)
  Phase 2 — ESTABLISHED : high jitter spread, thin_trail aktiv (timeout 60)

Behavioral Polyethism (v11) — Rollen-Verteilung über die Kolonie:
  50% Forward  : Phase 0 → 1 → 2 (klassisch, sammelt Erfahrung)
  30% Reverse  : Phase 2 → 1 → 0 (start vorsichtig, wird zum Experten-Scout)
  20% Stuck    : fest in einer Phase (zufällig 0, 1 oder 2)
  → Kolonie als Ganzes wird adaptiver. Wenn eine Strategie versagt
    (z.B. Quellenwechsel, Hindernis-Update), kompensieren andere Rollen.

Drain-Mechanismen (v5):
  - Recruitment beim Nestverlassen (sniff food_pheromone am Ausgang)
  - RETURN_TO_FOOD-Schwelle 70% statt 95% → mehr Repeat-Trips
  - Spread-Drosselung bei starkem Food-Trail → bleib auf produktivem Pfad

Loop-Destruction (v6):
  - Loop-Trap Detection in EXPLORE: 3 Zellen mit beiden Pheromonen + Rest leer

Wall-Escape (v7):
  - Nach Wandanstoß in EXPLORE: Jitter min 80%, Nest-Avoidance 2.5x stärker

Loop-Escape (v8):
  - chase_ticks Counter: 60-100 Ticks Trail-Timeout je nach Phase
  - recruit_cooldown 30 Ticks nach Timeout
  - Visit-Window 80 in Trail-States für große Loops

Heim-Optimierung (v9):
  - Heim-Gradient Override: klar dominantes Heim-Pheromone gewinnt immer
  - Mixed-Pheromone Felder beim Tragen 4x unattraktiver
  - Anti-Reversal Filter schützt starke Heim-Trail-Nachbarn beim Tragen

Quellen-Optimierung (v10) — symmetrisch zum Heimweg:
  - Food-Gradient Override + Mixed-Cell Penalty + Direction-Smoothing
  - Outbound- und Inbound-Trails separieren sich räumlich über Trips

Biologisch plausibel:
  - Nur Pheromone-Wahrnehmung (lokal, im Nachbar-Radius)
  - Bewegungsvektor dx/dy = Körperausrichtung (Path Integration, wie Wüstenameise)
  - recent[] = kurzes Bewegungsgedächtnis (lokal, ~150 Schritte)
  - trips_completed = Erfahrungs-Counter (Lernkurve, kein Positions-GPS)
  - role = angeborene Verhaltenstendenz (biologisches Polyethism)

Keine globalen Koordinaten, keine gespeicherten Pfade.
"""

import math
import random

from ant_sim.models import (
    Action, ActionResult, AntAgent, DropAction, MoveAction,
    Perception, PickUpAction, Position, NeighborInfo,
)

class MyAgent(AntAgent):

    LEAVE_NEST, EXPLORE, CHASE_FOOD, RETURN_HOME, RETURN_TO_FOOD, STARVE = range(6)
    DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    PHERO_WINDOW = 50

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = self.LEAVE_NEST
        self.dx = self.dy = 0
        self.max_e = self.energy
        self.recent = []
        self.last_pos = None
        self.prev_pos = None
        self.max_phero = 1.0
        self.jitter = 0.45
        self.phero_samples = []
        self.bounce_count = 0
        self.escape_ticks = 0
        self.desperate = False
        self.thin_trail = 0

        # ── Erfahrungs-Tracking (kein Positionswissen) ──
        self.trips_completed = 0           # successful food drops at nest
        self.trail_commit = 0              # consecutive ticks on strong nest trail
        # ── Loop-Schutz: Trail-Abandonment + Recruit-Cooldown ──
        self.chase_ticks = 0               # ticks in CHASE_FOOD/RETURN_TO_FOOD
        self.recruit_cooldown = 0          # ticks während keine Rekrutierung erlaubt

        # ── Behavioral Polyethism: Rollen-Zuteilung für Phasen-Zyklus ──
        # 50% Forward (klassisch Scout → Builder → Established)
        # 30% Reverse (start Established → wird Scout, "senior expert" Karriere)
        # 20% Stuck (fest in zufälliger Phase — perpetueller Scout/Builder/Established)
        # Erzeugt Verhaltensdiversität in der Kolonie → robuster gegen Umgebungsänderungen
        r = random.random()
        if r < 0.20:
            self.role = ('stuck', random.randint(0, 2))
        elif r < 0.50:
            self.role = ('reverse', None)
        else:
            self.role = ('forward', None)

    # ═══════════════════════════════════════════════════════════
    #  Phase logic — drives all adaptive parameters
    # ═══════════════════════════════════════════════════════════

    def _phase(self):
        """Aktuelle Verhaltensphase — abhängig von Rolle und trips_completed.

        Forward (50%):  Scout(0) → Builder(1) → Established(2)
        Reverse (30%):  Established(2) → Builder(1) → Scout(0)
        Stuck   (20%):  fest in einer Phase (0, 1 oder 2)
        """
        role_type, role_data = self.role
        if role_type == 'stuck':
            return role_data
        if role_type == 'reverse':
            if self.trips_completed >= 3:
                return 0
            if self.trips_completed >= 1:
                return 1
            return 2
        # Forward (default)
        if self.trips_completed == 0:
            return 0
        if self.trips_completed < 3:
            return 1
        return 2

    def _blend_alpha(self):
        # phase 0: 0.6 (responsive, Agent 8). phase 2: 0.7 (stable, Agent 14).
        return 0.6 + 0.05 * min(self._phase(), 2)

    def _chase_timeout(self):
        """Trail-Abandonment Schwelle — späte Phase ist ungeduldiger,
        weil die typische Trip-Länge bekannt ist."""
        return 100 - 20 * min(self._phase(), 2)   # phase 0: 100, phase 2: 60

    # ═══════════════════════════════════════════════════════════
    #  decide
    # ═══════════════════════════════════════════════════════════

    def decide(self, perception: Perception) -> Action:
        p = perception

        # ── Position tracking (lokal, kurzes Gedächtnis) ──
        self.prev_pos = self.last_pos
        self.last_pos = (p.current_x, p.current_y)
        self.recent.append((p.current_x, p.current_y))
        if len(self.recent) > 150:
            self.recent = self.recent[-80:]

        if p.current_is_nest:
            self.max_e = max(self.max_e, p.energy)
            self.bounce_count = 0

        # ── Bounce tracking ──
        if self.prev_pos == (p.current_x, p.current_y):
            self.bounce_count += 1
        elif self.prev_pos:
            self.bounce_count = max(0, self.bounce_count - 1)

        # ── Adaptive direction blending (Körperausrichtung) ──
        if self.prev_pos:
            mdx = p.current_x - self.prev_pos[0]
            mdy = p.current_y - self.prev_pos[1]
            if mdx != 0 or mdy != 0:
                a = self._blend_alpha()
                self.dx = self.dx * a + mdx * (1 - a)
                self.dy = self.dy * a + mdy * (1 - a)

        # ── Stuck detector ──
        if len(self.recent) >= 5:
            if (len(set(self.recent[-5:])) <= 3 or
                    len(self.recent) >= 10 and len(set(self.recent[-10:])) <= 4):
                self.dx, self.dy = random.choice(self.DIRS)

        # ── Pheromone perception (lokal) ──
        cur_phero = p.current_food_pheromone + p.current_nest_pheromone
        self._max_food_nb = p.current_food_pheromone
        self._max_nest_nb = p.current_nest_pheromone
        for n in p.neighbors:
            combo = n.food_pheromone + n.nest_pheromone
            if combo > cur_phero:
                cur_phero = combo
            if n.food_pheromone > self._max_food_nb:
                self._max_food_nb = n.food_pheromone
            if n.nest_pheromone > self._max_nest_nb:
                self._max_nest_nb = n.nest_pheromone

        self.phero_samples.append(cur_phero)
        if len(self.phero_samples) > self.PHERO_WINDOW:
            self.phero_samples = self.phero_samples[-self.PHERO_WINDOW:]
        self.max_phero = max(self.max_phero, cur_phero, 1.0)
        if p.current_is_nest or (p.current_has_food and not p.carries):
            if len(self.phero_samples) >= 3:
                mid = sorted(self.phero_samples)[len(self.phero_samples) // 2]
                self.max_phero = max(mid, 1.0)

        ratio = min(cur_phero / self.max_phero, 1.0)
        self._set_jitter(ratio)

        # ── Trail-Timer + Recruit-Cooldown ──
        if self.state in (self.CHASE_FOOD, self.RETURN_TO_FOOD) and not p.carries:
            self.chase_ticks += 1
        else:
            self.chase_ticks = 0
        if self.recruit_cooldown > 0:
            self.recruit_cooldown -= 1

        # ── Trail commitment (lokales Sensing): wie lange am starken Trail? ──
        if p.carries and self._max_nest_nb > self.max_phero * 0.3:
            self.trail_commit = min(self.trail_commit + 1, 10)
        else:
            self.trail_commit = max(self.trail_commit - 1, 0)

        # ── Desperate mode — erst ab Phase 1 (Scout darf nicht panisch sein) ──
        if self._phase() >= 1:
            self.desperate = self.max_e > 0 and p.energy <= self.max_e * 0.15
            if self.desperate:
                self.max_phero = 1.0
                self._flee_vec()
                if self._max_food_nb > 0.5 and self._max_nest_nb > 0.5:
                    self.jitter = 1.5 * 2.00
                else:
                    self.jitter = 1.5 * 1.20
        else:
            self.desperate = False

        if self.escape_ticks > 0:
            self.escape_ticks -= 1
            self.jitter = 1.5 * 1.80

        # ── Pick up food ──
        if p.current_has_food and not p.carries:
            self._flip()                       # body turns around
            self.state = self.RETURN_HOME
            self.recent.clear()
            self.trail_commit = 0
            return PickUpAction(p.current_x, p.current_y)

        if p.carries and self.state not in (self.RETURN_HOME,):
            self.state = self.RETURN_HOME
        if not p.carries and self.state == self.RETURN_HOME:
            self.state = self.EXPLORE

        return [
            self._leave_nest, self._explore, self._chase_food,
            self._return_home, self._return_to_food, self._starve,
        ][self.state](p)

    # ═══════════════════════════════════════════════════════════
    #  Adaptive jitter — phase-dependent
    # ═══════════════════════════════════════════════════════════

    def _set_jitter(self, ratio):
        phase = self._phase()
        mp = self.max_phero if self.max_phero > 0 else 1.0
        food_r = min(self._max_food_nb / mp, 1.0)
        nest_r = min(self._max_nest_nb / mp, 1.0)

        if self.state == self.EXPLORE:
            # phase 0: 30..150% (Agent 8 — geradeaus für schnellen Trail-Aufbau)
            # phase 1: 42..150% (Übergang)
            # phase 2: 55..150% (Agent 14 — breite Streuung für sparse food)
            base = 0.30 + 0.125 * min(phase, 2)
            # DRAIN MODE: starker Food-Trail in Sicht → Spread drosseln, Trail folgen
            if food_r > 0.3:
                base = 0.30
            # WALL BOUNCE: nach Wandanstoß stark erhöhter Jitter zum Ausbruch
            # (verhindert dass die Ameise zurück Richtung Nest gedrückt wird)
            if self.bounce_count >= 1:
                base = max(base, 0.80)
            self.jitter = 1.5 * (base + (1.50 - base) * ratio)

        elif self.state == self.RETURN_HOME:
            if nest_r > 0.3:
                self.jitter = 1.5 * 0.12              # starker Trail → minimal jitter
            elif food_r > 0.3 and nest_r < 0.1:
                self.jitter = 1.5 * 0.55              # food zone, escape
            elif food_r < 0.05 and nest_r < 0.05:
                self.jitter = self._lost_jitter()
            else:
                self.jitter = 1.5 * 0.20

        else:  # CHASE_FOOD, RETURN_TO_FOOD, STARVE
            self.jitter = 1.5 * (0.05 + 0.25 * ratio)        # 5..30%

    def _lost_jitter(self):
        """Carrying food, no pheromones: trust direction vector (Wüstenameise)."""
        if self.bounce_count > 5:
            return 1.5 * 0.65
        if self.bounce_count > 2:
            return 1.5 * 0.45
        # No walls, no trail: vertraue dem geflippten direction vector
        return 1.5 * 0.25

    def _flee_vec(self):
        """Lokaler Mittelwert recent positions als Flucht-Zentroid."""
        if len(self.recent) < 3:
            return
        cx = sum(pos[0] for pos in self.recent) / len(self.recent)
        cy = sum(pos[1] for pos in self.recent) / len(self.recent)
        fx, fy = self.x - cx, self.y - cy
        mag = (fx * fx + fy * fy) ** 0.5
        if mag > 0.1:
            fx, fy = fx / mag, fy / mag
            self.dx = self.dx * 0.7 + fx * 0.3
            self.dy = self.dy * 0.7 + fy * 0.3

    # ═══════════════════════════════════════════════════════════
    #  Helpers
    # ═══════════════════════════════════════════════════════════

    def _acc(self, p):
        return [n for n in p.neighbors if n.is_accessible]

    def _vis(self):
        s = {(pos.x, pos.y) for pos in self.memory}
        s.update(self.recent)
        return s

    def _dot(self, n, p):
        return self.dx * (n.x - p.current_x) + self.dy * (n.y - p.current_y)

    def _cos(self, n):
        ox, oy = n.x - self.x, n.y - self.y
        d_sq = self.dx * self.dx + self.dy * self.dy
        o_sq = ox * ox + oy * oy
        if d_sq < 0.01 or o_sq < 0.01:
            return 0.0
        return (self.dx * ox + self.dy * oy) / (d_sq ** 0.5 * o_sq ** 0.5)

    def _best(self, acc, score_fn):
        if self.carries:
            for n in acc:
                if n.is_nest:
                    return n

        # Soft anti-reversal — aber starke Pheromon-Nachbarn schützen
        # (sonst filtert der Algorithmus den eigentlichen Trail raus wenn er rückwärts ist)
        d_sq = self.dx * self.dx + self.dy * self.dy
        if d_sq > 0.01 and len(acc) > 1:
            worst_n = min(acc, key=lambda n: self._cos(n))
            if self._cos(worst_n) < -0.5:
                # Beim Tragen: starke Heim-Pheromone schützen (Richtung Heim)
                # Beim Suchen: starke Food-Pheromone schützen (Richtung Quelle)
                going_to_food = (not self.carries and
                                 self.state in (self.CHASE_FOOD, self.RETURN_TO_FOOD))
                protect = ((self.carries and worst_n.nest_pheromone > 0.5) or
                           (going_to_food and worst_n.food_pheromone > 0.5))
                if not protect:
                    acc = [n for n in acc if n is not worst_n]

        prev = self.prev_pos
        freq = {}
        # Trail-States: längeres Window (80) erkennt große Loops (30-40+ Zellen)
        # EXPLORE/STARVE: kürzeres Window für schnelle Lokal-Variation
        if self.state in (self.CHASE_FOOD, self.RETURN_TO_FOOD):
            window = 80
        elif self._phase() == 0:
            window = 24
        else:
            window = 30
        for pos in self.recent[-window:]:
            freq[pos] = freq.get(pos, 0) + 1

        vis = self._vis() if self.desperate else None
        scored = []
        for n in acc:
            s = score_fn(n) + random.random() * self.jitter
            if prev and (n.x, n.y) == prev:
                s -= 50.0
            s -= n.agent_count * 3.0
            if self._cos(n) < -0.7:
                s -= self.max_phero * 3.0
            vc = freq.get((n.x, n.y), 0)
            penalty = 4.0 if self._phase() == 0 else 6.0
            s -= vc * vc * penalty
            if self.desperate and vis:
                if (n.x, n.y) in vis:
                    s -= 30.0
                if n.food_pheromone > 0 and n.nest_pheromone > 0:
                    s -= 25.0
                s += self._cos(n) * 1.0
            scored.append((s, n))
        return max(scored, key=lambda t: t[0])[1]

    def _go(self, n):
        return MoveAction(n.x, n.y)

    def _flip(self):
        self.dx, self.dy = -self.dx, -self.dy

    def _fallback(self, p):
        return MoveAction(p.neighbors[0].x, p.neighbors[0].y)

    def _sniff_food_trail(self, p):
        """Recruitment: find strongest food_pheromone neighbor. None if no trail."""
        acc = self._acc(p)
        food_nbs = [n for n in acc if n.food_pheromone > 0]
        if not food_nbs:
            return None
        return max(food_nbs, key=lambda n: n.food_pheromone)

    def _loop_trap_detected(self, p):
        """Loop-Signatur: genau 3 Zellen (current + Nachbarn) tragen BEIDE
        Pheromone, alle übrigen sind leer. Heißt: andere Ameisen kreiseln
        hier ohne echte Quelle. Nicht reingehen, auch wenn's nach Food riecht.
        """
        mp = self.max_phero if self.max_phero > 0 else 1.0
        both_thresh = mp * 0.15      # "aktiv beide" — nicht nur Rauschen
        empty_thresh = mp * 0.05     # "praktisch leer"

        cells = [(p.current_food_pheromone, p.current_nest_pheromone)]
        cells += [(n.food_pheromone, n.nest_pheromone) for n in p.neighbors]

        both = sum(1 for f, ne in cells if f > both_thresh and ne > both_thresh)
        empty = sum(1 for f, ne in cells if f < empty_thresh and ne < empty_thresh)

        # Pattern: exakt 3 mit beiden, Rest fast komplett leer
        return both == 3 and empty >= len(cells) - 3

    def _gradient_override(self, p, acc, target_attr, opposite_attr):
        """Findet klar dominanten reinen Pheromone-Nachbarn (target_attr stark,
        opposite_attr schwach). Returns Neighbor oder None wenn kein klarer Gewinner.

        Symmetrisch nutzbar für Heim- und Food-Trail. Pheromone-Gradient dominiert
        IMMER über Path Integration wenn ein klares Signal vorliegt.
        """
        mp = self.max_phero if self.max_phero > 0 else 1.0
        mixed_thresh = mp * 0.10
        pure = [n for n in acc
                if getattr(n, target_attr) > 0
                and getattr(n, opposite_attr) < mixed_thresh]
        if not pure:
            return None
        best = max(pure, key=lambda n: getattr(n, target_attr))
        others = [getattr(n, target_attr) for n in acc if n is not best]
        second = max(others) if others else 0
        # Klarer Gewinner: 1.8x zweitstärkstes + 0.5 floor (kein Rauschen)
        if getattr(best, target_attr) > second * 1.8 + 0.5:
            return best
        return None

    # ═══════════════════════════════════════════════════════════
    #  LEAVE_NEST — Recruitment: am Ausgang nach Food-Trail schnüffeln
    # ═══════════════════════════════════════════════════════════

    def _leave_nest(self, p):
        self.recent.clear()
        self.bounce_count = 0
        self.chase_ticks = 0
        # RECRUITMENT: stärkstes Food-Pheromone → CHASE_FOOD direkt
        # (außer Recruit-Cooldown aktiv — Ameise hatte gerade einen Loop-Timeout)
        if self.recruit_cooldown == 0:
            trail = self._sniff_food_trail(p)
            if trail:
                self.dx = trail.x - p.current_x
                self.dy = trail.y - p.current_y
                self.state = self.CHASE_FOOD
                return self._chase_food(p)
        # Kein Trail (oder Cooldown) → zufällige Scout-Richtung
        self.dx, self.dy = random.choice(self.DIRS)
        self.state = self.EXPLORE
        return self._explore(p)

    # ═══════════════════════════════════════════════════════════
    #  EXPLORE — phase-adaptive jitter, scout phase = geradeaus
    # ═══════════════════════════════════════════════════════════

    def _explore(self, p):
        acc = self._acc(p)
        if not acc:
            return self._fallback(p)

        for n in acc:
            if n.has_food:
                return self._go(n)

        # LOOP TRAP: nur Explorer — bei 3-Zellen-Mixed-Pattern Food-Smell ignorieren
        loop_trap = self._loop_trap_detected(p)

        if any(n.food_pheromone > 0 for n in acc) and not loop_trap:
            self.state = self.CHASE_FOOD
            return self._chase_food(p)

        # Phase 0 hält länger durch (40%) — sucht aus ohne Heim-Panik
        starve_thresh = 0.40 if self._phase() == 0 else 0.50
        if self.max_e > 0 and p.energy <= self.max_e * starve_thresh:
            self._flip()
            self.state = self.STARVE
            return self._starve(p)

        vis = self._vis()
        mp = self.max_phero if self.max_phero > 0 else 1.0
        trap_thresh = mp * 0.15
        # Nest-Vermeidung verstärkt nach Wandanstoß
        nest_avoid = 1.5 if self.bounce_count >= 1 else 0.6

        def score(n):
            # Phase 0: starkes direction-weight → geradeaus für schnellen Trail-Aufbau
            dot_w = 2.0 if self._phase() == 0 else 1.2
            s = self._dot(n, p) * dot_w
            if (n.x, n.y) in vis:
                s -= 11.0
            # Nest-Pheromone aktiv meiden — verhindert "Heim-Drift" beim Erkunden
            s -= n.nest_pheromone * nest_avoid
            # LOOP TRAP: aktiv weg von Mixed-Pheromone Zellen
            if loop_trap and n.food_pheromone > trap_thresh and n.nest_pheromone > trap_thresh:
                s -= 100.0
            return s

        return self._go(self._best(acc, score))

    # ═══════════════════════════════════════════════════════════
    #  CHASE_FOOD — thin_trail nur in Phase 2 (Builder darf nicht abbrechen)
    # ═══════════════════════════════════════════════════════════

    def _chase_food(self, p):
        acc = self._acc(p)
        if not acc:
            return self._fallback(p)

        for n in acc:
            if n.has_food:
                self.chase_ticks = 0
                return self._go(n)

        if self.max_e > 0 and p.energy <= self.max_e * 0.25:
            self._flip()
            self.state = self.STARVE
            return self._starve(p)

        # ── TRAIL ABANDONMENT: zu lange auf Trail ohne Food → Loop-Verdacht ──
        # Recruit-Cooldown setzen damit Ameise nicht direkt wieder auf den
        # gleichen Trail rekrutiert wird
        if self.chase_ticks > self._chase_timeout() and self._phase() >= 1:
            self.chase_ticks = 0
            self.recruit_cooldown = 30
            self._flip()
            self.state = self.STARVE
            return self._starve(p)

        # Erst ab Phase 2 (etablierte Kolonie) — vorher braucht's jeden Trail
        if self._phase() >= 2:
            food_nbs = sum(1 for n in acc if n.food_pheromone > 0)
            peak = max((n.food_pheromone for n in acc), default=0)
            if food_nbs == 1 and peak < self.max_phero * 0.15:
                self.thin_trail += 1
            else:
                self.thin_trail = 0
            if (self.max_e > 0 and p.energy <= self.max_e * 0.80
                    and self.thin_trail >= 8):
                self.thin_trail = 0
                self._flip()
                self.state = self.STARVE
                return self._starve(p)

        if not p.current_food_pheromone and all(n.food_pheromone == 0 for n in acc):
            self.state = self.EXPLORE
            return self._explore(p)

        def score(n):
            s = 0.0
            if n.food_pheromone > 0:
                s += 50.0
            s += n.food_pheromone * 5.0
            if n.has_food:
                s += 200.0
            s += self._dot(n, p) * 0.5
            return s

        return self._go(self._best(acc, score))

    # ═══════════════════════════════════════════════════════════
    #  RETURN_HOME — Pheromone-Gradient + Path Integration via dx/dy
    # ═══════════════════════════════════════════════════════════

    def _return_home(self, p):
        if p.current_is_nest and p.carries:
            self._flip()
            self.trips_completed += 1          # ← Erfahrung wächst, Phase steigt
            # Lowered threshold 95→70: mehr Repeat-Trips zur bekannten Quelle
            recruit_thresh = 0.70 if self._phase() >= 1 else 0.85
            if p.energy >= self.max_e * recruit_thresh:
                self.state = self.RETURN_TO_FOOD
                self.escape_ticks = 5
            else:
                self.state = self.EXPLORE
            self.recent.clear()
            self.trail_commit = 0
            return DropAction(p.current_x, p.current_y)

        acc = self._acc(p)
        if not acc:
            return self._fallback(p)

        mp = self.max_phero if self.max_phero > 0 else 1.0
        mixed_thresh = mp * 0.10                  # Schwelle für "Mixed-Feld"

        # ── HEIM-GRADIENT OVERRIDE ──
        # Wenn ein reines Heim-Feld klar stärker ist als alle anderen Nachbarn,
        # dann dorthin — egal wohin der Körper zeigt. Pheromone-Gradient dominiert
        # IMMER über Path Integration.
        override = self._gradient_override(p, acc, 'nest_pheromone', 'food_pheromone')
        if override:
            self.dx = override.x - p.current_x
            self.dy = override.y - p.current_y
            self.trail_commit = 0
            return self._go(override)

        vis = self._vis()
        no_trail = self._max_nest_nb < 0.1 and self._max_food_nb < 0.1
        committed = self.trail_commit >= 3

        def score(n):
            # Mixed-Pheromone Feld 4x unattraktiver — bevorzuge reinen Heim-Trail
            # (Ameise folgt dem klaren Heimweg statt der Mischzone wo beide Trails kreuzen)
            is_mixed = n.food_pheromone > mixed_thresh
            nest_weight = 1.25 if is_mixed else 5.0       # 5.0 / 4 = 1.25
            s = n.nest_pheromone * nest_weight
            # Heim-Vektor (Path Integration via dx/dy) — Junction-Entscheidung
            # Bei mehreren forward-Nachbarn entscheidet der interne Vektor + Jitter
            s += self._dot(n, p) * (3.0 if no_trail else 1.8)
            if n.is_nest:
                s += 200.0
            # Trail commitment: bestrafe Schritte gegen die Bewegungsrichtung stärker
            # ABER nur wenn das Reverse-Feld nicht selbst starkes Heim-Pheromone hat
            if committed and self._cos(n) < -0.3 and n.nest_pheromone < 1.0:
                s -= 25.0
            if (n.x, n.y) in vis:
                s -= 5.0
            if n.food_pheromone > 0 and n.nest_pheromone == 0 and not n.is_nest:
                s -= n.food_pheromone * 0.8
            if n.nest_pheromone == 0 and not n.is_nest:
                s -= 10.0
            return s

        return self._go(self._best(acc, score))

    # ═══════════════════════════════════════════════════════════
    #  RETURN_TO_FOOD
    # ═══════════════════════════════════════════════════════════

    def _return_to_food(self, p):
        acc = self._acc(p)
        if not acc:
            return self._fallback(p)

        for n in acc:
            if n.has_food:
                self.chase_ticks = 0
                return self._go(n)

        if self.max_e > 0 and p.energy <= self.max_e * 0.25:
            self._flip()
            self.state = self.STARVE
            return self._starve(p)

        # ── TRAIL ABANDONMENT (gleich wie CHASE_FOOD) ──
        if self.chase_ticks > self._chase_timeout() and self._phase() >= 1:
            self.chase_ticks = 0
            self.recruit_cooldown = 30
            self._flip()
            self.state = self.STARVE
            return self._starve(p)

        if not p.current_food_pheromone and all(n.food_pheromone == 0 for n in acc):
            self.state = self.EXPLORE
            return self._explore(p)

        mp = self.max_phero if self.max_phero > 0 else 1.0
        mixed_thresh = mp * 0.10

        # ── FOOD-GRADIENT OVERRIDE (symmetrisch zu RETURN_HOME) ──
        # Reinstes Food-Pheromon klar stärker als alle anderen → direkt hin.
        # Sucht den effizientesten Outbound-Trail zur Quelle.
        override = self._gradient_override(p, acc, 'food_pheromone', 'nest_pheromone')
        if override:
            self.dx = override.x - p.current_x
            self.dy = override.y - p.current_y
            self.trail_commit = 0
            return self._go(override)

        def score(n):
            # Mixed-Pheromone Feld 4x unattraktiver — bevorzuge reinen Food-Trail
            # (Outbound-Route, sauber, ohne Inbound-Verkehr)
            is_mixed = n.nest_pheromone > mixed_thresh
            food_weight = 1.25 if is_mixed else 5.0       # 4x weniger bei mixed
            s = 0.0
            if n.food_pheromone > 0:
                s += 50.0
            s += n.food_pheromone * food_weight
            if n.has_food:
                s += 200.0
            # Heim-Pheromone aktiv meiden (zeigt zurück zum Nest, falsche Richtung)
            s -= n.nest_pheromone * 0.5
            # Stärkeres Direction-Smoothing für Pfad-Optimierung
            # (kantenglättung — Ameisen schneiden über Trips natürlich Kurven ab)
            s += self._dot(n, p) * 1.8                    # was 1.0
            return s

        return self._go(self._best(acc, score))

    # ═══════════════════════════════════════════════════════════
    #  STARVE — Pheromone-Gradient + direction vector
    # ═══════════════════════════════════════════════════════════

    def _starve(self, p):
        if p.current_is_nest:
            self.state = self.LEAVE_NEST
            return self._leave_nest(p)

        acc = self._acc(p)
        if not acc:
            return self._fallback(p)

        no_trail = self._max_nest_nb < 0.1

        def score(n):
            s = n.nest_pheromone * 4.0
            s += self._dot(n, p) * (2.5 if no_trail else 1.0)
            if n.is_nest:
                s += 200.0
            return s

        return self._go(self._best(acc, score))


DefaultAgent = MyAgent
