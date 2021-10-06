"""
Python implementation of DiscreteProbabilityDistribution class


@author aloaberasturi

"""

import math
import json
import numpy as np

class ProbabilityDistribution:

    def __init__(self, support, probability):
        self.support = support
        self.time_stamp = 0

        # try:
        #     if probability == None:
        #         self.probability = {sample : 1.0/len(self.support)  if self.is_uniform else 0.0 for sample in self.support}
        
        # except ValueError:
            # if len(probability) != 0:
        self.probability = {sample: p for sample, p in zip(self.support, probability)}
                
        self.calculate_min_max_support_values()


    def calculate_min_max_support_values(self):
        try:
            self.max_support_value = max(self.support)
            self.min_support_value = min(self.support)
        except ValueError:
            print("The support of the distribution is empty")
    
    def put(self, sample, prob):
    # We have a discrete probability distribution, 
    # so we map the sample to the corresponding slot

        sample = self.map_sample(sample)
        self.probability[sample] = prob
    
    def map_sample(self, sample):
        if (sample < self.min_support_value * 2 or sample > self.max_support_value * 2):
            raise ValueError(
                            """Error mapping a sample in the discrete probability distribution.
                            Samples cannot be out of the range: %d and %d" % 
                            (self.min_support_value * 2, self.max_support_value * 2) """
                        )
        if (sample < self.min_support_value):
            return self.min_support_value
        
        if (sample > self.max_support_value):
            return self.max_support_value
        
        return round(sample)

    def get_support(self):
        return self.support

    def get_prob(self, sample):
        sample = self.map_sample(sample)

        if (sample in self.probability.keys()):
            return self.probability[sample]
        return -1

    def get_probs(self):        
        probs = [self.probability[sample] for sample in self.get_support()]
        return probs

    def get_time_stamp(self): return self.time_stamp

    def set_time_stamp(self, time_stamp) : self.time_stamp = time_stamp

    def check_integrity(self):
        sum = 0.0
        for prob in self.probability.values():
            if (prob < 0.0 or prob > 1.0):
                return False
            sum += prob
        
        return abs(1.0 - sum) < 0.00000001

    def mre(self, q):
        """
        Checks whether the sum of the constraints exceeds 1 and normalize it	

        Parameters:
        -----------
        q : ProbabilityConstraint

        """

        if (q.check_integrity() == False):
            q.normalize()

        sum_constraint = 0.0
        sum_current_prob = 0.0
        constraint_keys = q.keys() 

        for key in constraint_keys:
            sum_constraint += q[key]
            sum_current_prob += self.probability[key]
        
        normalizer = (1.0 - sum_constraint) / (1.0 - sum_current_prob) if sum_current_prob < 1 else 0.0

        for sample in self.support:
            prob_value = q[sample] if sample in constraint_keys else normalizer * self.probability[sample]
            self.probability[sample] = prob_value


    def decay(self, distribution, t_prime, t, t_max, nu): 

        grace_period = t_max

        #change names : t == t_prime, tn == t, pace == t_max == omega, rate = nu, 
        delta_t = t_prime - t
        delta_t = 0 if delta_t < grace_period else 1 + (delta_t / t_max) 

        new_prob = 0.0
        for sample in self.support:
            new_prob = distribution.get_prob(sample) + ( (self.probability[sample] - distribution.get_prob(sample) ) * math.pow(nu, delta_t) )
            self.probability[sample] = new_prob

    def normalize(self):
        pass

    def entropy(self):
        entropy = 0.0
        for p in self.probability.values():
            if (p != 0.0):
                entropy += p * math.log(p)

        return -entropy

    def get_max_value(self):
        max_prob = 0.0
        for sample in self.support:
            if self.get_prob(sample) > max_prob:
                max_prob = self.get_prob(sample)
                result_sample = sample

        return result_sample

    def __str__(self):
        """
        Overrides __str__() method
        """
        string = "Probability distribution: " + json.dumps(self.probability)
        return string
   

class ProbabilityConstraints():
    """
    A class for constraints in probability distributions. Constraints is a dictionary
    """
    def __init__(self, constraints):
        self.constraints = constraints
    
    def __getitem__(self, key):
        return self.constraints.__getitem__(key)
    
    def __setitem__(self, key, val):
        self.constraints.__setitem__(key, val)

    def keys(self):
        return self.constraints.keys()

    def values(self):
        return self.constraints.values()

    def check_integrity(self):
        sum = 0.0
        for prob in self.values():
            sum += prob

        return sum <= 1

    def normalize(self):
        sum = 0.0
        for prob in self.values():
            sum += prob
        
        new_prob = 0.0
        for k in self.keys():
            new_prob = self.constraints[k] / sum
            self.constraints[k] = new_prob


    