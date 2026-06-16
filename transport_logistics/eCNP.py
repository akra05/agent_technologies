from dataclasses import dataclass
import threading 
import queue
import json

@dataclass(frozen=True)
class Message:
    msg_type: str   # 'CFP', 'PROPOSE', 'REFUSE', 'REJECT', 'ACCEPT' // nochmal umbenennen
    sender: object
    information: dict   # which resource is asked for and general task informations

class WarehouseAgent:
    """Lageragent (LA) — Initiator im CNP, schreibt seine Items aus."""

    def __init__(self, information):
        self.msg_queue = queue.Queue()
        self.information = information
        self.available_items = list(information['items'])  # Kopie der verfügbaren Items

class TransportAgent:
    def __init__(self, information):
        self.msg_queue = queue.Queue()
        self.information = information
        self.contracted_items = {}   # item -> LA
        self.pending_accepts = {}    # item -> LA (vorläufig)
        self.rejected_warehouses = {}  # item -> [LA]
        self.planned_route = []      # [(item, LA), ...]

    def plan_initial_route(self, warehouse_agents):
        self.planned_route = []
        current_pos = self.information['position']
        for item in self.information['requested_items']:
            excluded = self.rejected_warehouses.get(item, [])
            nearest_la = self._find_nearest(item, current_pos, warehouse_agents, excluded)
            if nearest_la is None:
                print(f"{self.information['id']}: kein Lager für Item {item} gefunden!")
                continue
            self.planned_route.append((item, nearest_la))
            current_pos = nearest_la.information['position']

    def send_initial_bids(self):
        current_pos = self.information['position']
        for item, la in self.planned_route:
            distance = self._distance(current_pos, la.information['position'])
            la.msg_queue.put(Message(
                msg_type='PRELIMINARY_BID',
                sender=self,
                information={
                    'item': item,
                    'distance': distance
                }
            ))
            current_pos = la.information['position']

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
        return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])

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
        })
        transport_agents.append(ta)

    run_simulation(transport_agents,warehouse_agents)

if __name__ == "__main__":
    main()