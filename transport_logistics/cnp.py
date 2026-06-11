from dataclasses import dataclass
import threading 
import queue

NUMBER_OF_THREADS = 10

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
        for i in self.participants:
            i.msg_queue.put(f"Hi are you reading: {i.information}")
        return

class Participant:
    def __init__(self,information):
        self.msg_queue = queue.Queue()
        self.information = information

    def run(self):
        while True:
            msg = self.msg_queue.get()
            print(msg)
            break
        return



def main():

    threads = []
    participants = []
    

    for i in range(NUMBER_OF_THREADS):
        participant = Participant({"position":(4,4),"thread_id":i})
        participants.append(participant)
        t = threading.Thread(target=participant.run)
        threads.append(t)
    
    initiator = Initiator({"position":(3,3)},participants)
    t = threading.Thread(target=initiator.run)
    threads.append(t)

    for t in threads[:-1]:
        t.start()
    
    threads[-1].start()

    for t in threads:
        t.join()

if __name__ == "__main__":
    main()

    


    

    