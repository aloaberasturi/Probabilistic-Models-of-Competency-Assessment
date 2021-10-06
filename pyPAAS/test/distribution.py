"""
Pythonic implementation of ProbabilityDistribution test

@author aloaberasturi

"""

from PAAS import utils
from PAAS.distribution import ProbabilityDistribution


def distribution_test():
    d1 = ProbabilityDistribution(utils.compute_support(), is_uniform=True)
    d2 = ProbabilityDistribution(utils.compute_support(), is_uniform=True)
    d3 = utils.product_operator(d1, d2)
    print('Probability distribution 1: ' + d1.__str__())
    print('Probability distribution 2: ' + d2.__str__())
    print('Product of d1 times d2: ' + d3.__str__())
    print('Distribution 3 entropy: ' + str(d3.entropy()))


distribution_test()
