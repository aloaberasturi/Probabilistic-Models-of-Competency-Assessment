"""

Python serviceEigentrust implementation

@author aloaberasturi

"""
from copy import copy
import sys
import numpy as np
import PAAS.common as common
from PAAS.exception import PeerAssessmentException
from PAAS.distribution import ProbabilityDistribution, ProbabilityConstraints
from PAAS.assessment import Assessment
from PAAS.peer import Peer

class ServiceEigentrust:
    def __init__(self, num_students, informed_prior):
        self.submissions = []
        self.assessments = {}
        self.peer_assessments = {}
        self.leader_assessments = {}
        self.leader = None
        self.informed_prior = informed_prior
        self.is_directly_trusted = [None] * num_students # Element i represents if the tutor has a direct trust in student i
        # initialize T and C with ignorance probability distributions
        self.d = common.get_ignorance_difference_distribution(self.informed_prior)
        self.T = np.array([ copy(self.d) for i in range(num_students)]) # teacher vector
        self.C = np.array([[copy(self.d) for j in range(num_students)] for i in range(num_students)]) # peers matrix

    def set_leader(self, leader):
        self.leader = leader
        
    @staticmethod
    def add_assessment_to_dict(d, submission, assessment):

        """
        Adds an assessment to a dictionary

        ...

        Parameters
        ----------
        d : dict
        submission : Submission
        assessment : Assessment
        """
        if (submission not in d.keys()):
            d[submission] = [assessment]
        else:
            d[submission].append(assessment)

    def delete_assessment(self, assessment):
        referee    = assessment.get_referee()
        submission = assessment.get_submission()
        mark       = assessment.get_mark()
        time_stamp = assessment.get_time_stamp()   

        if self.is_already_assessed(submission, referee):
            for stored_assessment in self.assessments[submission]:
                prev_submission = stored_assessment.get_submission()
                prev_referee = stored_assessment.get_referee()
                if (prev_submission.__eq__(submission) and prev_referee.__eq__(referee)):
                    self.assessments[submission].remove(stored_assessment)
                    self.leader_assessments.pop(submission)
                    break


    def add_peer_assessment(self, new_assessment):
        referee    = new_assessment.get_referee()
        index1     = referee.get_index()
        submission = new_assessment.get_submission()
        mark       = new_assessment.get_mark()
        time_stamp = new_assessment.get_time_stamp()

        if (self.is_already_assessed(submission, referee)):
            raise PeerAssessmentException(f'Error: {referee.__str__()} has already assessed {submission.__str__()}')

        if (referee.is_leader):

            # Update leader or raise exception if another leader was already defined

            if (self.leader is None):
                self.leader = referee
            elif (not self.leader.__eq__(referee)):
                raise PeerAssessmentException("Error: Can't define two different leaders in the same experiment")

        # Check for common assessments to calculate direct trust distributions

        # 1) Update C matrix
        if (submission in self.assessments.keys()):
            for previous_assessment in self.assessments[submission]:
                referee2 = previous_assessment.get_referee()
                index2   = referee2.get_index()


                # Calculate the evaluation difference between the assessents of r1 and r2
                # If negative, r1 under rates with respect to r2
                # If positive, r1 over rates with respect to r2
                # If zero, r1 and r2 give the same evaluation

                evaluation_difference_r1r2 = common.calculate_difference( mark, previous_assessment.get_mark() )
                evaluation_difference_r2r1 = - evaluation_difference_r1r2

                # Update the trust distribution in the C matrix or T vector accordingly (Ecs. 2 & 3)

                trust_distribution_r1r2 = None
                trust_distribution_r2r1 = None

                if (referee.is_leader):
                    trust_distribution_r1r2 = self.T[index2] # we are not interested in the trust that students have in the teacher
                    self.is_directly_trusted[index2] = True
                elif (referee2.is_leader):
                    trust_distribution_r2r1 = self.T[index1] # we are not interested in the trust that students have in the teacher
                    self.is_directly_trusted[index1] = True

                else:
                    trust_distribution_r1r2 = self.C[index1][index2]
                    trust_distribution_r2r1 = self.C[index2][index1]

                if (trust_distribution_r1r2 is not None):
                    self.increase_point_in_probability_distribution(trust_distribution_r1r2, time_stamp, evaluation_difference_r1r2)

                if (trust_distribution_r2r1 is not None):
                    self.increase_point_in_probability_distribution(trust_distribution_r2r1, time_stamp, evaluation_difference_r2r1)

            # Compute indirect trust
            # 2) Iterate t_k+1 = C'*t_k until |t_k+1 - t_k| to calculate the tutor's indirect trust

            error = sys.maxsize
            T_next = [None] * len(self.T)
            while True:
                for j in range(len(self.T)):
                    if (self.is_directly_trusted[j]):

                        # the trust distribution remains the same
                        T_next[j] = self.T[j]
                        continue
                    for i in range(len(self.T)):
                        if (i == j) : continue
                        if (T_next[j] is None): T_next[j] = common.product_operator(self.C[i][j], self.T[i])
                        else :
                            T_next[j] = common.argmin_EMD(T_next[j], common.product_operator(self.C[i][j], self.T[i]))

                error = common.calculate_error(self.T, T_next)
                self.T = T_next

                if (error < common.MAX_ERROR):
                    break

        # 3) Store submission and assessment

        if (submission not in self.submissions):
            self.submissions.append(submission)

        ServiceEigentrust.add_assessment_to_dict(self.assessments, submission, new_assessment)

        if (referee.is_leader):
            self.leader_assessments[submission] = new_assessment
        else:
            ServiceEigentrust.add_assessment_to_dict(self.peer_assessments, submission, new_assessment)


    def increase_point_in_probability_distribution(self, distribution, decay, sample):

        # decay the trust distribution. The first time T is modified, t is the
        # timestamp of the evaluation involved in the modification.

        distribution.decay(self.d, decay, distribution.get_time_stamp(), common.T_MAX, common.T_MAX)
        distribution.set_time_stamp(decay)

        # update the trust distribution between r1 and r2
        sample = distribution.map_sample(sample)

        # update the probability of having this evaluation rate between r1 and r2 increasing the prob
        prev_prob = distribution.get_prob(sample)
        prob = prev_prob + common.GAMMA * (1 - prev_prob)
        new_point = ProbabilityConstraints({sample : prob})
        distribution.mre(new_point)

    def calculate_joint_entropy(self, submissions):
        probabilities = []
        for submission in submissions:
            # If there´s a tutor assessment, this one is returned and converted into a probability distribution

            if submission in self.leader_assessments.keys():
                mark = self.leader_assessments[submission].mark
                probabilities.append(common.get_delta_distribution(mark)) 

            # Otherwise, an automated assessment is calculated
            # as an aggregation of peers opinions from the perspective of the teacher

            else :
                peermarks = self.peer_assessments[submission]
                if len(peermarks) > 0:
                    probabilities.append(self.calculate_leader_mark_distribution(submission, peermarks))

        entropies = np.array([p.entropy() for p in probabilities])
        return entropies.sum()
        for p in probabilities:
            a = p.entropy()
        # product = probabilities.pop(0)
        # for p in probabilities:
        #     product = common.product_operator(product, p) 

        # return product
            

    def calculate_automated_marks(self):
        result = []
        for submission in self.submissions:

            # If there´s a tutor assessment, this one is returned

            if submission in self.leader_assessments.keys():
                result.append(self.leader_assessments[submission])

            # Otherwise, an automated assessment is calculated
            # as an aggregation of peers opinions from the perspective of the teacher

            else :
                peermarks = self.peer_assessments[submission]
                if len(peermarks) > 0:
                    leader_mark_distribution = self.calculate_leader_mark_distribution(submission, peermarks)
                    uncertainty = leader_mark_distribution.entropy()
                    automatic_peer = Peer("AUTOMATED_ASSESSMENT")
                    assessment = Assessment(automatic_peer, submission, leader_mark_distribution.get_max_value(), 'AUTOMATED_ASSESSMENT', -1, uncertainty)
                    result.append(assessment)

        return result


    def calculate_leader_mark_probability_given_one_opinion(self, mark, assessment):

        """
        Implements Service_Eigentrust.calculateTutorMarkProbabilityGivenOpinion()
        (Corresponds to eq. [10])

        Parameters
        ----------
        mark : double
        assessment : Assessment

        Returns
        -------
        scalar
               Probability value
        """

        def compute_sum(interval):
            addition = 0
            for i in interval:
                addition += trust_distribution_tutor_r.get_prob(i)
            return addition

        index = assessment.get_referee().get_index()
        trust_distribution_tutor_r = self.T[index]

        if (mark == 0):
            n = common.calculate_difference(0, assessment.get_mark())
            interval = np.arange(-common.MAX_MARK_VALUE, n+1)
            return compute_sum(interval)

        elif (mark == common.MAX_MARK_VALUE):
            n = common.calculate_difference(common.MAX_MARK_VALUE, assessment.get_mark())
            interval = np.arange(n, common.MAX_MARK_VALUE + 1)
            return compute_sum(interval)

        difference = common.calculate_difference(mark, assessment.get_mark())
        return trust_distribution_tutor_r.get_prob(difference)



    def calculate_leader_mark_probability_given_many_opinions(self, submission, mark, assessments):
        """
        Implements Service_Eigentrust.calculateTutorMarkProbabilityGivenManyOpinions()
        (Corresponds to eq. [11])

        Parameters
        ----------
        submission : Submission
        mark : double
        assessments : Assessment array

        Returns
        -------
        scalar
            Probability value
        """

        aggregation = 0

        def aggregate(m, assessment):
            nonlocal aggregation
            index = assessment.get_referee().get_index()
            information = common.distributions_EMD(self.T[index], self.d)
            prob = self.calculate_leader_mark_probability_given_one_opinion(m, assessment) * information
            aggregation *= (1 - prob)
            aggregation += prob
            return information

        sum = 0
        for i in range(common.MAX_MARK_VALUE + 1):
            aggregation = 0
            for assessment in assessments:
                aggregate(i, assessment)

            sum += aggregation

        aggregation = 0
        information_sum = 0
        for assessment in assessments:
            information_sum += aggregate(mark, assessment)

        if (information_sum <= common.DELTA): # opinions do not contain enough information
            return common.get_initial_marking_distribution(self.informed_prior).get_prob(mark)
        return aggregation / sum

    def calculate_leader_mark_distribution(self, submission, opinions):

        """
        Implements Service_Eigentrust.calculateTutorEvaluationDistributionGivenOpinions()

        Parameters
        ----------

        submission : Submission
        opinions : array of assessments

        Returns
        -------

        ProbabilityDistribution
                The probability distribution of the leader's marks
        """

        probs = np.array([self.calculate_leader_mark_probability_given_many_opinions(submission, i, opinions)
                for i in range(common.MAX_MARK_VALUE + 1)])

        return ProbabilityDistribution(common.get_marking_values(), probability = probs)

    def is_already_assessed(self, submission, referee):

        """
        Implements former Service_Eigentrust.containsAssessment() by Patricia
        Checks if a submission has already been assessed by a given referee


        Parameters
        ----------
        submission : Submission
        referee : Peer

        Returns
        -------
        Bool
        """
        if (submission not in self.assessments.keys()):
            return False

        for stored_assessment in self.assessments[submission]:
            prev_submission = stored_assessment.get_submission()
            prev_referee = stored_assessment.get_referee()
            if (prev_submission.__eq__(submission) and prev_referee.__eq__(referee)):
                return True

    def get_C(self) : return self.C

    def get_T(self) : return self.T

    def set_C(self, C) : self.C = C

    def set_T(self, T) : self.T = T


