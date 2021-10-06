"""
Python implementation of Peer class

@author aloaberasturi

"""

class Peer:
    def __init__(self, name, index = -1):
        self.name = name
        self.index = index            
        self.is_leader = False
        self.profile = {}

    # Setters

    def set_index(self, index)           : self.index = index
    def set_as_leader(self, is_leader)   : self.is_leader = is_leader
    def set_mark(self, submission, mark) : self.profile[submission] = mark

    # Getters

    def get_name(self) : return self.name
    def get_index(self): return self.index
    def get_mark(self, submission) : return self.profile[submission]


    def __str__(self):
        """
        Overrides __str__() method
        """
        string = f'Referee:  {self.get_name()}'
        return string
    
    def __eq__(self, object):
        """
        Overrides __eq__() method
        """
        if (not isinstance(object, Peer)) : return False
        return object.__str__().__eq__(self.__str__())
        
    def __hash__(self):
        """
        Overrides __hash__() method
        """
        return self.__str__().__hash__()