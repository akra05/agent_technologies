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
}
"""
@dataclass(frozen=True)
class Message:
    msg_type: str   # 'CFP', 'PROPOSE', 'REFUSE', 'REJECT', 'ACCEPT'
    sender: object
    content: dict   # which resource is asked for and general task informations

class Initiator:
    def __init__(self,information,participants):
        self.msg_queue = queue.Queue()
        self.information = information
        self.participants = participants

    def run(self):
        self.constraint_function()
        for i in self.participants:
            msg = Message('CFP', self, self.information)
            i.msg_queue.put(msg)
        return
    
    
        
class Participant:
    def __init__(self,information):
        self.msg_queue = queue.Queue()
        self.information = information

    def run(self):
        while True:
            msg = self.msg_queue.get()
            
            break
        return

def constraint_function(participant,msg):
        
        if not (participant.information['position'][0] >= 0 and participant.information['position'][0] <= msg.information['world_size'] and self.information['position'][1] >= 0 and self.information['position'][1] <= msg.information['world_size']):
            return False # if Participant is outside the world return False which continues in a 'REFUSE' message
        if not msg.information['item'] in participant.information['items']:
            return False # if Participant doesn't have the needed item return False which continues in a 'REFUSE' message
        return True

def main():

    threads = []
    participants = []
    

    for i in range(NUMBER_OF_THREADS):
        participant = Participant({"position":(4,4),"items":['A','A','B'],"thread_id":i})
        participants.append(participant)
        t = threading.Thread(target=participant.run)
        threads.append(t)
    
    initiator = Initiator({"position":(3,3), "items":['A','A','B'],"world_size":10},participants)
    t = threading.Thread(target=initiator.run)
    threads.append(t)

    for t in threads[:-1]:
        t.start()
    
    threads[-1].start()

    for t in threads:
        t.join()

if __name__ == "__main__":
    main()

    


    

    