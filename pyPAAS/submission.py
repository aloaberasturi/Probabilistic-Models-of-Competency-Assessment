"""

Python implementation of Submission class

@author aloaberasturi

"""

from PAAS.peer import Peer

class Submission:
    """
    A class used to represent a submission

    ...

    Attributes
    ----------
    name : str
    author : str
    """
    def __init__(self, ID, author = 'defaultAuthor'):
        self.ID = ID
        self.author = author

    # getters

    def get_ID(self) : return self.ID
    def get_author(self) : return self.author.get_name()
    def get_rank(self): return self.rank

    # setters

    def set_rank(self, rank) : self.rank = rank


    def __lt__(self, other):
        """
        Check if self has a higher rank than other
        """
        if (isinstance(other, Submission)):
            return self.get_rank() < other.get_rank()
        return NotImplemented

    def __str__(self):
        return f'Submission {self.get_ID()} by {self.author.__str__()}'

    def __hash__(self):
        return self.__str__().__hash__()

    def __eq__(self, object):
        """
        Overrides __eq__() method
        """
        if (not isinstance(object, Submission)): return False
        return object.__str__().__eq__(self.__str__())

