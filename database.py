import pickle
from typing import Dict
from person import Person

class Database:
    def __init__(self, filename: str):
        self.filename = filename
        self.people: Dict[str, Person] = {}
        self.load_people()

    def load_people(self):
        try:
            with open(self.filename, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    self.people = data
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            self.people = {}

    def save_people(self):
        with open(self.filename, 'wb') as f:
            pickle.dump(self.people, f)

    def get_person(self, key: str) -> Person:
        return self.people.get(key)

    def add_person(self, key: str, person: Person):
        self.people[key] = person

    def remove_person(self, key: str):
        if key in self.people:
            del self.people[key]

