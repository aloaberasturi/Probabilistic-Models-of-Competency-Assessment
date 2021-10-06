"""
Python implementation of Assessment class

@author aloaberasturi

"""

class Assessment:

    """
    A class used to represent an assessment

    ...

    Attributes
    ----------
        referee : Peer
        submission : Submission
        mark : double
        assessment_type : str
        time_stamp : double
        uncertainty : double
    """
    def __init__(self, referee, submission, mark, assessment_type, time_stamp, uncertainty = 0.0):

        self.submission = submission
        self.referee = referee
        self.mark = mark
        self.assessment_type = assessment_type
        self.time_stamp = time_stamp
        self.uncertainty = uncertainty    

    # Getters

    def get_submission(self) : return self.submission
    def get_referee(self)    : return self.referee
    def get_type(self)       : return self.assessment_type
    def get_mark(self)       : return self.mark
    def get_time_stamp(self) : return self.time_stamp
    def get_uncertainty(self): return self.uncertainty

    # Setters

    def set_mark(self, mark) : self.mark = mark
    def set_time_stamp(self, time_stamp) : self.time_stamp = time_stamp
    def set_uncertainty(self, uncertainty) : self.uncertainty = uncertainty

    def __str__(self):
        """
        Overrides __str__() method
        """
        string = f'{self.referee.__str__()}, {self.submission.__str__()}:   < Mark: {str(self.mark)} > Assessment type: {self.assessment_type}'
        # string = self.referee.__str__() + ", " + self.submission.__str__() + ":      <"
        # string += "Mark: " + str(self.mark) + ","
        # string = string[:len(string) - 1]
        # string += ">    " + "Assessment type: " + self.assessment_type
        self.str = string

        return string


    def __eq__(self, object):
        """
        Overrides __eq__() method
        """
        if (not isinstance(object, Assessment)): return False

        return object.__str__().__eq__(self.__str__())
    
    



    
