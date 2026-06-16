from dataclasses import dataclass
import threading 
import queue

NUMBER_OF_THREADS = 10

"""
information(Initiator): 
{
position: (3,3) Tupel
resource_type: ['A','A','B'] list
world_size: 10 int
}
"""
"""
content_dict(Message):
{
position(Sender)
requested_items/available_items(Sender)
evaluation
}
"""
@dataclass(frozen=True)
class Message:
    msg_type: str   # 'CFP', 'PROPOSE', 'REFUSE', 'REJECT', 'ACCEPT'
    sender: object
    information: dict   # which resource is asked for and general task informations

class Initiator:
    def __init__(self,information,participants):
        self.msg_queue = queue.Queue()
        self.information = information
        self.participants = participants

    def run(self):
        
        for i in self.participants:
            msg = Message('CFP', self, self.information)
            i.msg_queue.put(msg)

        proposals = []
        msg_counter = 0

        while True:
            msg = self.msg_queue.get()
            
            msg_counter += 1

            if msg.msg_type == 'PROPOSE':
                proposals.append(msg)
            
            if msg_counter == len(self.participants):
                break
        
        if not proposals:
            print("Kein Participant konnte die Aufgabe übernehmen!")
            return

        sorted_proposals = sorted(proposals, key=lambda msg: msg.information['value'])
        winner = sorted_proposals[0]

        msg = Message('ACCEPT',self,self.information)
        winner.sender.msg_queue.put(msg)

        for proposal in sorted_proposals[1:]:  # alles außer winner
            msg = Message('REJECT', self, self.information)
            proposal.sender.msg_queue.put(msg)

        return
    
    
        
class Participant:
    def __init__(self,information):
        self.msg_queue = queue.Queue()
        self.information = information

    def run(self):
        while True:
            msg = self.msg_queue.get()
            if msg.msg_type == 'CFP':

                if self.constraint_function(msg):
                    return_msg = Message('PROPOSE',self,{'value':self.evaluation_function(msg)})
                    msg.sender.msg_queue.put(return_msg)
                else:
                    return_msg = Message('REFUSE',self,{})
                    msg.sender.msg_queue.put(return_msg)
                    break
                
            if msg.msg_type == 'ACCEPT':
                print("Gewonnen!")
                self.information['items'].remove(msg.information['needed_item'])
                print(self.information)
                break  # jetzt fertig
            
            if msg.msg_type == 'REJECT':
                print("Verloren!")
                break  # jetzt fertig     
    
    def constraint_function(self,msg):
        
        if not (self.information['position'][0] >= 0 and self.information['position'][0] <= msg.information['world_size'] and self.information['position'][1] >= 0 and self.information['position'][1] <= msg.information['world_size']):
            return False # if Participant is outside the world return False which continues in a 'REFUSE' message
        if not msg.information['needed_item'] in self.information['items']:
            return False # if Participant doesn't have the needed item return False which continues in a 'REFUSE' message
        return True
    
    def evaluation_function(self,msg):
        return ((abs(self.information['position'][0]-msg.information['position'][0]))**2+(abs(self.information['position'][1]-msg.information['position'][1]))**2)**0.5

def main():

    threads = []
    participants = []
    

    for i in range(NUMBER_OF_THREADS):
        participant = Participant({"position":(4,4),"items":['A','A','B'],"thread_id":i})
        participants.append(participant)
        t = threading.Thread(target=participant.run)
        threads.append(t)
    
    initiator = Initiator({"position":(4,3), "needed_item":'A',"world_size":10},participants)
    t = threading.Thread(target=initiator.run)
    threads.append(t)

    for t in threads[:-1]:
        t.start()
    
    threads[-1].start()

    for t in threads:
        t.join()

if __name__ == "__main__":
    main()

    


    

    