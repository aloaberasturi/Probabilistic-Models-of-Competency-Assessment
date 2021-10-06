"""
A custom error module 

@author aloaberasturi

"""

class NonNormalizedDistributionError(Exception):
    """
    Raised when attempting to build a non-normalized distribution
    """
    pass

class PeerAssessmentException(Exception):
    """
    Raised when trying to repeat an assessment by the same person or 
    when trying to create more than one leader
    """
    pass