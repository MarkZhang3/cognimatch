import uuid 

class Person:
    """
    stores the user inputted information about themselves ie.
    info: dictionary storing fields like: { extrovertness: 0-10, ..., hobbies: [...], ...}
    histories: stores log of this person's interactions with other persons that have been simulated
    compatibility: dict storing other person and compability score between this person to that person 
    """
    def __init__(self, person_id: str, name: str, user_info: dict = None):
        self.name = name
        self.id = person_id if person_id is not None else str(uuid.uuid4())
        self.info = user_info if user_info is not None else {}

        self.histories = []       
        self.compatibilities = {} 

    def add_history(self, partner_name: str, conversation_log: str, timestamp: str = None):
        entry = {
            "partner": partner_name,
            "log": conversation_log,
            "timestamp": timestamp
        }
        self.histories.append(entry)

    def set_compatibility(self, partner_name: str, score: float):
        self.compatibilities[partner_name] = score
    
    def __repr__(self):
        return f"Person(name={self.name}, info={self.info}, histories={len(self.histories)})"

    