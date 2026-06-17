from dataclasses import dataclass
#import threading
import queue
import json


@dataclass(frozen=True)
class Message:
    msg_type: str  # 'CFP', 'PROPOSE', 'REFUSE', 'REJECT', 'ACCEPT' // nochmal umbenennen
    sender: object
    information: dict  # which resource is asked for and general task informations


class WarehouseAgent:
    """Lageragent (LA) — Initiator im eCNP, schreibt seine Items aus."""

    def __init__(self, information):
        self.msg_queue = queue.Queue()
        self.information = information
        self.available_items = list(information['items'])
        self.bids = {}  # item -> [{'ta': ta, 'distance': d}]
        self.done = False

    def collect_definitive_bids(self):
        self.definitive_bids = {}
        while not self.msg_queue.empty():
            msg = self.msg_queue.get()
            if msg.msg_type == 'DEFINITIVE_BID':
                item = msg.information['item']
                if item not in self.definitive_bids:
                    self.definitive_bids[item] = []
                self.definitive_bids[item].append({
                    'ta': msg.sender,
                    'distance': msg.information['distance']
                })


    def send_definitive_responses(self):
        """Sendet DEFINITIVE_ACCEPT an besten Bieter, DEFINITIVE_REJECT an alle anderen."""
        if self.done:
            return

        for item, bidders in self.definitive_bids.items():
            if not bidders:
                continue
            if item not in self.available_items:
                # Item bereits vergeben — alle ablehnen
                for bidder in bidders:
                    print(f"  DEFINITIVE_REJECT → {bidder['ta'].information['id']} (Item '{item}' bereits vergeben)")
                    bidder['ta'].msg_queue.put(Message(
                        msg_type='DEFINITIVE_REJECT',
                        sender=self,
                        information={'item': item}
                    ))
                continue

            best = min(bidders, key=lambda b: b['distance'])
            print(f"\n[eCNP] {self.information['id']} Item '{item}':")
            print(f"  DEFINITIVE_ACCEPT → {best['ta'].information['id']} (Distanz: {best['distance']})")

            best['ta'].msg_queue.put(Message(
                msg_type='DEFINITIVE_ACCEPT',
                sender=self,
                information={
                    'item': item,
                    'distance': best['distance']
                }
            ))
            self.available_items.remove(item)  # Item als vergeben markieren

            for bidder in bidders:
                if bidder['ta'] is not best['ta']:
                    print(f"  DEFINITIVE_REJECT → {bidder['ta'].information['id']}")
                    bidder['ta'].msg_queue.put(Message(
                        msg_type='DEFINITIVE_REJECT',
                        sender=self,
                        information={'item': item}

                    ))


            if not self.available_items:
                self.done = True

    def collect_bids(self):
        """Sammelt alle eingehenden PRELIMINARY_BIDs."""
        if self.done:
            return

        while not self.msg_queue.empty():
            msg = self.msg_queue.get()
            if msg.msg_type == 'PRELIMINARY_BID':
                item = msg.information['item']
                if item not in self.bids:
                    self.bids[item] = []
                self.bids[item].append({
                    'ta': msg.sender,
                    'distance': msg.information['distance']
                })

    def send_preliminary_responses(self):
        """Sendet PRE_ACCEPT an ALLE besten Bieter, PRE_REJECT an alle anderen."""
        if self.done:
            return

        for item, bidders in self.bids.items():
            if not bidders:
                continue

            # 1. Den minimalen Abstand ermitteln
            min_dist = min(b['distance'] for b in bidders)

            # 2. Alle Bieter herausfiltern, die genau diesen Abstand haben
            best_bidders = [b for b in bidders if b['distance'] == min_dist]

            print(f"\n[eCNP] {self.information['id']} Item '{item}':")

            # 3. Schleife für ALLE Gewinner (bekommen alle ein PRE_ACCEPT)
            for bidder in best_bidders:
                print(f"  PRE_ACCEPT → {bidder['ta'].information['id']} (Distanz: {bidder['distance']})")
                bidder['ta'].msg_queue.put(Message(
                    msg_type='PRE_ACCEPT',
                    sender=self,
                    information={
                        'item': item,
                        'distance': bidder['distance']
                    }
                ))

            # 4. Schleife für die Verlierer (alle mit einer schlechteren Distanz)
            for bidder in bidders:
                if bidder['distance'] > min_dist:
                    print(f"  PRE_REJECT → {bidder['ta'].information['id']}")
                    bidder['ta'].msg_queue.put(Message(
                        msg_type='PRE_REJECT',
                        sender=self,
                        information={
                            'item': item,
                            'best_distance': min_dist
                        }
                    ))

        self.bids = {}  # Zurücksetzen für die nächste Runde
class TransportAgent:
    def __init__(self, information, warehouse_agents):
        self.msg_queue = queue.Queue()
        self.information = information
        self.warehouse_agents = warehouse_agents        # ← speichern
        self.contracted_items = {}  # item -> LA
        self.pending_accepts = {}  # item -> LA (vorläufig)
        self.rejected_warehouses = {}  # item -> [LA]
        self.temp_rejected = {}          # temporär (nach PRE_REJECT) ← neu
        self.planned_route = []  # [(item, LA), ...]
        self.give_up_items = set()  # Items für die keine Lösung möglich ist ← neu

    # in TransportAgent:
    def print_route(self):
        print(f"  {self.information['id']} ({self.information['position']}):")
        for item, la in self.contracted_items.items():
            print(f"    Item '{item}' ← {la.information['id']} (kontraktiert)")
        for item, la in self.planned_route:
            if item not in self.contracted_items:
                print(f"    Item '{item}' ← {la.information['id']} (geplant)")
        for item in self.give_up_items:
            print(f"    Item '{item}' → Konventionalstrafe")


    def plan_initial_route(self, warehouse_agents):
        self.planned_route = []
        current_pos = self.information['position']
        for item in self.information['requested_items']:
            if item in self.contracted_items or item in self.give_up_items:
                continue  # ← give_up_items auch überspringen!

            excluded = (self.rejected_warehouses.get(item, []) +
                        self.temp_rejected.get(item, []))

            nearest_la = self._find_nearest(item, current_pos, warehouse_agents, excluded)
            if nearest_la is None:
                self.temp_rejected[item] = []
                nearest_la = self._find_nearest(item, current_pos, warehouse_agents,
                                                self.rejected_warehouses.get(item, []))
            if nearest_la is None:
                # Kein Lager mehr → aufgeben!
                self.give_up_items.add(item)
                print(f"  {self.information['id']}: Item '{item}' → Konventionalstrafe")
                continue  # ← nicht return, damit andere Items noch geplant werden

            self.planned_route.append((item, nearest_la))
            current_pos = nearest_la.information['position']

    def send_initial_bids(self):
        current_pos = self.information['position']
        for item, la in self.planned_route:
            if item in self.contracted_items:
                # bereits kontraktiert → Position updaten, weitergehen
                current_pos = self.contracted_items[item].information['position']
                continue
            
            # Nur für das ERSTE nicht-kontraktierte Item bieten!
            distance = self._distance(current_pos, la.information['position'])
            la.msg_queue.put(Message(
                msg_type='PRELIMINARY_BID',
                sender=self,
                information={
                    'item': item,
                    'distance': distance
                }
            ))
            break  # ← nach erstem offenen Item aufhören!

    def _find_nearest(self, item, from_pos, warehouse_agents, exclude=None):
        if exclude is None:
            exclude = []
        candidates = [la for la in warehouse_agents
                      if item in la.available_items
                      and la not in exclude]
        if not candidates:
            return None
        return min(candidates, key=lambda la: self._distance(from_pos, la.information['position']))

    def _distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def process_definitive_responses(self):
        while not self.msg_queue.empty():
            msg = self.msg_queue.get()
            if msg.msg_type == 'DEFINITIVE_ACCEPT':
                item = msg.information['item']
                la = msg.sender
                self.contracted_items[item] = la
                self.pending_accepts.pop(item, None)
                self.temp_rejected.pop(item, None)
                print(f"  {self.information['id']} hat Item '{item}' von {la.information['id']} erhalten")

            elif msg.msg_type == 'DEFINITIVE_REJECT':
                item = msg.information['item']
                la = msg.sender
                if item not in self.rejected_warehouses:
                    self.rejected_warehouses[item] = []
                self.rejected_warehouses[item].append(la)
                self.temp_rejected[item] = []
                self.pending_accepts.pop(item, None)
                print(f"  {self.information['id']} dauerhaft abgelehnt für '{item}' bei {la.information['id']}")

        # Nächstes offenes Item suchen und direkt bieten
        missing = [i for i in self.information['requested_items']
                if i not in self.contracted_items and i not in self.give_up_items]
        
        if missing:
            self.plan_initial_route(self.warehouse_agents)
            self.send_initial_bids()

    def process_preliminary_responses(self):
        while not self.msg_queue.empty():
            msg = self.msg_queue.get()
            if msg.msg_type == 'PRE_ACCEPT':
                item = msg.information['item']
                self.pending_accepts[item] = (msg.sender, msg.information['distance'])
            elif msg.msg_type == 'PRE_REJECT':
                item = msg.information['item']
                if item not in self.temp_rejected:
                    self.temp_rejected[item] = []
                self.temp_rejected[item].append(msg.sender)

        # Nächstes offenes Item finden
        next_item = None
        for item in self.information['requested_items']:
            if item not in self.contracted_items and item not in self.give_up_items:
                next_item = item
                break
        
        if next_item is None:
            return  # alles erledigt

        if next_item in self.pending_accepts:
            # PRE_ACCEPT erhalten → DEFINITIVE_BID schicken
            la, distance = self.pending_accepts[next_item]
            print(f"  {self.information['id']} sendet DEFINITIVE_BID für '{next_item}' an {la.information['id']}")
            la.msg_queue.put(Message(
                msg_type='DEFINITIVE_BID',
                sender=self,
                information={'item': next_item, 'distance': distance}
            ))
        else:
            # PRE_REJECT → neu planen
            print(f"  {self.information['id']} replant für Item '{next_item}'")
            self.plan_initial_route(self.warehouse_agents)
            # Nur bieten wenn noch was zu planen ist
            if self.planned_route:
                self.send_initial_bids()


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def run_simulation(transport_agents, warehouse_agents):
    # Schritt 1: Alle LAs schicken gleichzeitig CFPs raus
    for la in warehouse_agents:
        for item in la.available_items:
            for ta in transport_agents:
                ta.msg_queue.put(Message(
                    msg_type='CFP',
                    sender=la,
                    information={
                        'item': item,
                        'position': la.information['position']
                    }
                ))

    # Schritt 2: Jeder TA verarbeitet seine CFPs und plant Route
    for ta in transport_agents:
        ta.plan_initial_route(warehouse_agents)
        ta.send_initial_bids()

    max_rounds = 10
    for round_nr in range(max_rounds):
        print(f"\n--- Runde {round_nr + 1} ---")

        # Routen ausgeben
        print("Aktuelle Routen:")
        for ta in transport_agents:
            ta.print_route()
        print()
        # Schritt 3: LAs sammeln Gebote und senden PRE_ACCEPT/PRE_REJECT  ← neu
        for la in warehouse_agents:
            la.collect_bids()
            la.send_preliminary_responses()

        # Schritt 4: TAs verarbeiten PRE_ACCEPT/PRE_REJECT
        for ta in transport_agents:
            ta.process_preliminary_responses()

            # Schritt 5: LAs sammeln DEFINITIVE_BIDs, senden DEFINITIVE_ACCEPT/REJECT  ← neu
        for la in warehouse_agents:
            la.collect_definitive_bids()
            la.send_definitive_responses()

            # Schritt 6: TAs verarbeiten DEFINITIVE_ACCEPT/REJECT, passen Route an
        for ta in transport_agents:
            ta.process_definitive_responses()

        # Abbruch wenn alle TAs alle Items haben
        if all(
                set(ta.contracted_items.keys()) | ta.give_up_items >= set(ta.information['requested_items'])
                for ta in transport_agents):
            print("Simulation beendet!")
            break


def main():
    config = load_config('2exp_config.json')

    world_size = config['world_size']

    warehouse_agents = []
    for wh in config['warehouses']:
        wa = WarehouseAgent({
            "id": wh['id'],
            "position": tuple(wh['position']),  # Liste → Tuple
            "items": wh['items']
        })
        warehouse_agents.append(wa)

    transport_agents = []
    for fc in config['factories']:
        ta = TransportAgent({
            "id": fc['id'],
            "position": tuple(fc['position']),
            "requested_items": fc['requested_items']
        },
         warehouse_agents
        )
        transport_agents.append(ta)

    run_simulation(transport_agents, warehouse_agents)

    # Endergebnis
    print(f"\n{'='*50}")
    print("ENDERGEBNIS")
    print(f"{'='*50}")
    
    total_distance = 0
    total_penalty = 0
    
    for ta in transport_agents:
        print(f"\n{ta.information['id']} ({ta.information['position']}):")
        
        ta_distance = 0
        prev_pos = ta.information['position']
        
        for item, la in ta.contracted_items.items():
            d = ta._distance(prev_pos, la.information['position'])
            ta_distance += d
            prev_pos = la.information['position']
            print(f"  Item '{item}' ← {la.information['id']} ({la.information['position']}) | Distanz: {d}")
        
        # Rückweg zur Zentrale
        d_back = ta._distance(prev_pos, ta.information['position'])
        ta_distance += d_back
        
        penalty = len(ta.give_up_items) * 100
        total_penalty += penalty
        total_distance += ta_distance
        
        print(f"  Rückweg zur Zentrale: {d_back}")
        print(f"  Rundreise-Distanz: {ta_distance}")
        
        if ta.give_up_items:
            print(f"  Nicht erfüllt: {ta.give_up_items} → Strafe: {penalty}")
    
    print(f"\nGesamtdistanz:      {total_distance}")
    print(f"Gesamtstrafe:       {total_penalty}")
    print(f"Gesamtenergie:      {total_distance + total_penalty}")


if __name__ == "__main__":
    main()