"""
Pythonic implementation of ServiceEigentrust test

@author aloaberasturi

"""

from PAAS.peer import Peer
from PAAS.submission import Submission
from PAAS.assessment import Assessment
from PAAS.eigentrust import ServiceEigentrust
from PAAS.assessment_type import AssessmentType

def service_eigentrust_test():
    leader = Peer('tutor')
    leader.is_leader = True
    dave = Peer('Dave', 0)
    patricia = Peer('Patricia', 1)
    bruno = Peer('Bruno', 2)

    ex1 = Submission('ex1')
    ex2 = Submission('ex2')
    ex3 = Submission('ex3')
    ex4 = Submission('ex4')
    ex5 = Submission('ex5')
    ex6 = Submission('ex6')
    ex7 = Submission('ex7')
    ex8 = Submission('ex8')

    service = ServiceEigentrust(3)

    service.add_peer_assessment(Assessment(leader, ex1, 2, AssessmentType.TUTOR_ASSESSMENT.name, 1))
    service.add_peer_assessment(Assessment(dave, ex1, 1, AssessmentType.PEER_ASSESSMENT.name, 2))
    service.add_peer_assessment(Assessment(leader, ex2, 2, AssessmentType.TUTOR_ASSESSMENT.name, 3))
    service.add_peer_assessment(Assessment(dave, ex2, 1, AssessmentType.PEER_ASSESSMENT.name, 4))
    service.add_peer_assessment(Assessment(leader, ex3, 3, AssessmentType.TUTOR_ASSESSMENT.name, 5))
    service.add_peer_assessment(Assessment(dave, ex3, 2, AssessmentType.PEER_ASSESSMENT.name, 6))
    service.add_peer_assessment(Assessment(dave, ex4, 1, AssessmentType.PEER_ASSESSMENT.name, 7))
    service.add_peer_assessment(Assessment(patricia, ex4, 2, AssessmentType.PEER_ASSESSMENT.name, 8))
    service.add_peer_assessment(Assessment(dave, ex5, 1, AssessmentType.PEER_ASSESSMENT.name, 9))
    service.add_peer_assessment(Assessment(patricia, ex5, 2, AssessmentType.PEER_ASSESSMENT.name, 10))
    service.add_peer_assessment(Assessment(dave, ex6, 1, AssessmentType.PEER_ASSESSMENT.name, 11))
    service.add_peer_assessment(Assessment(patricia, ex6, 2, AssessmentType.PEER_ASSESSMENT.name, 6))
    service.add_peer_assessment(Assessment(patricia, ex7, 3, AssessmentType.PEER_ASSESSMENT.name, 13))
    service.add_peer_assessment(Assessment(bruno, ex7, 0, AssessmentType.PEER_ASSESSMENT.name, 13))

    print('leader-dave: ' + service.get_T()[dave.get_index()].__str__())
    print('leader-patricia: ' + service.get_T()[patricia.get_index()].__str__())
    print('leader-bruno: ' + service.get_T()[bruno.get_index()].__str__())
    print('dave-patricia: ' + service.get_C()[dave.get_index()][patricia.get_index()].__str__())
    print('patricia-dave: ' + service.get_C()[patricia.get_index()][dave.get_index()].__str__())

    print('------------- Automated Assesssments ---------------')
    for assessment in service.calculate_automated_marks():
        print(assessment.__str__())

service_eigentrust_test()
