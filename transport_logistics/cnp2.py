import json
import queue
import random
from dataclasses import dataclass


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


@dataclass
class Message:
    msg_type: str  # 'CFP', 'PROPOSE', 'REFUSE', 'REJECT', 'ACCEPT', 'CONFIRM'
    sender: object
    content: dict


class TransportAgent:
    """Transportagent (TA) — Participant im CNP, bietet auf Items."""

    def __init__(self, information):
        self.msg_queue = queue.Queue()
        self.information = information
        self.contracted_items = {}  # item_type -> LA

    def step(self):
        while not self.msg_queue.empty():
            msg = self.msg_queue.get()
            if msg.msg_type == 'CFP':
                self._handle_cfp(msg)
            elif msg.msg_type == 'ACCEPT':
                self._handle_accept(msg)
            elif msg.msg_type == 'REJECT':
                self._handle_reject(msg)

    def _handle_cfp(self, msg):
        item = msg.content['item']

        # Nur bieten wenn TA dieses Item noch braucht
        if item not in self.information['requested_items']:
            self.msg_queue.put(Message('REFUSE', self, {'item': item}))
            # direkt in eigene Queue — wird von LA nicht gebraucht, daher an LA senden:
            return

        if item in self.contracted_items:
            # Item bereits kontraktiert, nicht nochmal bieten
            msg.sender.msg_queue.put(Message('REFUSE', self, {'item': item}))
            return

        # Distanz berechnen — berücksichtigt bereits kontraktierte Items (Pfadkosten)
        distance = self._calculate_bid_distance(msg.sender.information['position'])
        msg.sender.msg_queue.put(Message('PROPOSE', self, {
            'distance': distance,
            'item': item
        }))

    def _calculate_bid_distance(self, target_pos):
        """Berechnet Distanz vom TA zur Zielposition, über bereits kontraktierte Lager."""
        if not self.contracted_items:
            # Direkte Distanz von Zentrale zum Lager
            return (abs(self.information['position'][0] - target_pos[0]) +
                    abs(self.information['position'][1] - target_pos[1]))
        else:
            # Distanz vom letzten kontraktierten Lager zum neuen Lager
            last_la = list(self.contracted_items.values())[-1]
            last_pos = last_la.information['position']
            return (abs(last_pos[0] - target_pos[0]) +
                    abs(last_pos[1] - target_pos[1]))

    def _handle_accept(self, msg):
        item = msg.content['item']
        la = msg.sender
        self.contracted_items[item] = la
        print(f"  {self.information['id']} erhält Item '{item}' von {la.information['id']}")
        msg.sender.msg_queue.put(Message('CONFIRM', self, {'item': item}))

    def _handle_reject(self, msg):
        item = msg.content['item']
        print(f"  {self.information['id']} abgelehnt für Item '{item}'")


class WarehouseAgent:
    """Lageragent (LA) — Initiator im CNP, schreibt seine Items aus."""

    def __init__(self, information):
        self.msg_queue = queue.Queue()
        self.information = information
        self.available_items = list(information['items'])  # Kopie der verfügbaren Items

    def run_cfp(self, item_type, transport_agents):
        """Schritt 1: Sendet CFP für einen Itemtyp an alle TAs."""
        if item_type not in self.available_items:
            return False  # LA hat dieses Item nicht

        print(f"\n[CNP] {self.information['id']} schreibt Item '{item_type}' aus")
        for ta in transport_agents:
            ta.msg_queue.put(Message('CFP', self, {
                'item': item_type,
                'la_position': self.information['position']
            }))
        return True

    def collect_proposals(self, transport_agents):
        """Schritt 3: Sammelt Proposals von allen TAs ein."""
        proposals = []
        for _ in transport_agents:
            response = self.msg_queue.get()
            if response.msg_type == 'PROPOSE':
                proposals.append(response)
        return proposals

        """Schritt 4: Wählt besten Bieter, sendet ACCEPT/REJECT."""


    def select_best(self, proposals, item_type):
        if not proposals:
            print(f"  Kein Angebot für '{item_type}' bei {self.information['id']}")
            return

        min_distance = min(p.content['distance'] for p in proposals)
        best_proposals = [p for p in proposals if p.content['distance'] == min_distance]
        best = random.choice(best_proposals)  # <-- Tie-Breaking

        # ACCEPT an besten Bieter
        best.sender.msg_queue.put(Message('ACCEPT', self, {
            'item': item_type,
            'distance': best.content['distance']
        }))

        # REJECT an alle anderen
        for p in proposals:
            if p.sender is not best.sender:
                p.sender.msg_queue.put(Message('REJECT', self, {'item': item_type}))

        # Item als vergeben markieren
        self.available_items.remove(item_type)

    def collect_confirm(self):
        """Schritt 5: Wartet auf CONFIRM vom gewählten TA."""
        self.msg_queue.get()


def run_simulation(warehouse_agents, transport_agents):
    print("=" * 50)
    print("Starte CNP-Simulation")
    print("=" * 50)

    # Alle Itemtypen die vergeben werden müssen
    all_item_types = set()
    for la in warehouse_agents:
        for item in la.available_items:
            all_item_types.add(item)

    for item_type in sorted(all_item_types):
        print(f"\n{'=' * 50}")
        print(f"Runde für Itemtyp: {item_type}")
        print(f"{'=' * 50}")

        # Nur LAs die dieses Item haben
        active_las = [la for la in warehouse_agents if item_type in la.available_items]

        for la in active_las:
            # Schritt 1: CFP senden
            la.run_cfp(item_type, transport_agents)

            # Schritt 2: TAs verarbeiten CFP und antworten
            for ta in transport_agents:
                ta.step()

            # Schritt 3+4: LA sammelt Proposals und wählt besten TA
            proposals = la.collect_proposals(transport_agents)
            la.select_best(proposals, item_type)

            # Schritt 5: TAs verarbeiten ACCEPT/REJECT
            for ta in transport_agents:
                ta.step()

            # LA wartet auf CONFIRM
            if proposals:
                la.collect_confirm()

    # Ergebnis
    print(f"\n{'=' * 50}")
    print("Ergebnis:")
    print(f"{'=' * 50}")
    total_distance = 0
    for ta in transport_agents:
        print(f"\n{ta.information['id']} ({ta.information['position']}):")
        ta_distance = 0
        prev_pos = ta.information['position']
        for item, la in ta.contracted_items.items():
            d = (abs(prev_pos[0] - la.information['position'][0]) +
                 abs(prev_pos[1] - la.information['position'][1]))
            ta_distance += d
            prev_pos = la.information['position']
            print(f"  Item '{item}' <- {la.information['id']} ({la.information['position']})")
        # Rückweg zur Zentrale
        d_back = (abs(prev_pos[0] - ta.information['position'][0]) +
                  abs(prev_pos[1] - ta.information['position'][1]))
        ta_distance += d_back
        print(f"  Rundreise-Distanz: {ta_distance}")
        total_distance += ta_distance

        missing = [i for i in ta.information['requested_items'] if i not in ta.contracted_items]
        if missing:
            penalty = len(missing) * 100
            print(f"  NICHT erfüllt: {missing} → Konventionalstrafe: {penalty} Energieeinheiten")
            total_distance += penalty

    print(f"\nGesamtenergieverbrauch: {total_distance}")


def main():
    config = load_config('2exp_config.json')
    world_size = config['world_size']

    warehouse_agents = []
    for wh in config['warehouses']:
        wa = WarehouseAgent({
            "id": wh['id'],
            "position": tuple(wh['position']),
            "items": wh['items'],
            "world_size": world_size
        })
        warehouse_agents.append(wa)

    transport_agents = []
    for fc in config['factories']:
        ta = TransportAgent({
            "id": fc['id'],
            "position": tuple(fc['position']),
            "requested_items": fc['requested_items'],
            "world_size": world_size
        })
        transport_agents.append(ta)

    run_simulation(warehouse_agents, transport_agents)
    print("\nDONE")


if __name__ == "__main__":
    main()