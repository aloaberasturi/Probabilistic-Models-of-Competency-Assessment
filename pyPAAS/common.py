"""
Python implementation of Utils class

@author aloaberasturi

"""

import scipy.stats as ss
import math
import numpy as np
import matplotlib.pyplot as plt
from PAAS.distribution import ProbabilityDistribution
from PAAS.exception import NonNormalizedDistributionError
from PAAS.filemanager import FileManager as FM
import pickle

# ************************* CONSTANTS ************************* #

MAX_ERROR = 0.1
MAX_MARK_VALUE = 3 # The maximum marking value in a range [0, MAX_MARK_VALUE].
GAMMA = 0.35
T_MAX = 1 # pace
NU = 0.98 # rate
DELTA = 0
ideal_difference = None

# ************************************************************* #

def get_initial_marking_distribution(informed_prior):
    """
    Returns an initial probability distribution over the grading differences.

    Parameters
    ----------
    informed_prior: bool
        indicates if the probability distribution over the grades is flat or not

    Returns
    -------
    ProbabilityDistribution
    """

    support = np.arange(0, MAX_MARK_VALUE+1)
    loc = support.min()
    max = support.max()
    scale = max-loc


    if not informed_prior:
        probs = ss.uniform(loc=loc, scale=scale).pdf(support)
    else: 
        probs = ss.norm.pdf(support, scale=0.7, loc=support.mean()) 
    
    probs /= probs.sum()
    x1 = np.random.choice(support, size=1000, p=probs)
    x2 = np.random.choice(support, size=1000, p=probs)
    x = x1-x2
    h,_ = np.histogram(x, bins=len(compute_support()))
    h = h / x.shape[0]
    plt.clf()
    plt.hist(x, bins=len(compute_support()))
    plt.savefig(FM().results_folder/'histogram')
    return ProbabilityDistribution(compute_support(), probability=h)

def get_delta_distribution(non_zero_value):
    """
    Returns a delta distribution with non-zero-probability argument given by non_zero_value
    """
    support = np.arange(0, MAX_MARK_VALUE + 1)
    probs = [0] * len(support)
    probs[non_zero_value] = 1.0
    return ProbabilityDistribution(support, probs)


def get_ignorance_difference_distribution(informed_prior=False):
    """
    Implements former Utils.getIgnoranceMarkingDifferenceDistribution() by Patricia


    Parameters
    ----------

    informed_priors: bool

    Returns
    -------
    ProbabilityDistribution
        Initial indirect trust distribution

    """

    d1 = get_initial_marking_distribution(informed_prior)
    d2 = get_initial_marking_distribution(informed_prior)
    ignorance_difference_distribution = product_operator(d1, d2)

    return ignorance_difference_distribution


def get_ideal_difference_distribution():
    """
    Implements former Utils.getIdealMarkingDifferenceDistribution() by Patricia

    Returns
    -------
    ProbabilityDistribution
                            Initial indirect trust distribution

    """

    global ideal_difference

    if (ideal_difference is None):
        support = compute_support()
        probs = np.array([1 if support[i] == 0 else 0 for i in range(len(support))])
        ideal_difference = ProbabilityDistribution(support, probability=probs)

    return ideal_difference


def get_marking_values():
    """Implements former Utils.getMarkingValues by Patricia

    Returns
    -------
    list
        List of possible marking values
    """
    array = np.arange(0, MAX_MARK_VALUE +1).tolist()
    return array

def compute_support():
    """
    Implements former Utils.getMarkingDifferenceValues by Patricia

    Returns
    -------
    list
        Support values of the probability distributions
    """
    # support_values = [i for i in range(-MAX_MARK_VALUE, MAX_MARK_VALUE + 1, 0.01)]

    return np.arange(-MAX_MARK_VALUE, MAX_MARK_VALUE + 1).tolist()

def product_operator(d1, d2):
    """
    Implements former Utils.productOperator() by Patricia

    Parameters
    ----------
    d1: Probability Distribution
    d2: Probability Distribution

    Returns
    -------
    ProbabilityDistribution
                            Product of d1 times d2
    """
    probs = [0] * len(compute_support())
    result = ProbabilityDistribution(compute_support(), probs)
    prob = [d1.get_prob(s1) * d2.get_prob(s2) for s1 in d1.get_support() for s2 in d2.get_support()]
    result_sample = [s1 + s2 for s1 in d1.get_support() for s2 in d2.get_support()]
    [result.put(rs, result.get_prob(rs) + p) for (rs, p) in zip(result_sample, prob)]

    if (not result.check_integrity()):
        raise NonNormalizedDistributionError("The aggregate operation generated a non normalized distribution")

    return result

def calculate_difference(mark1, mark2):
    """
    Implements former Utils.calculateDifference() by Patricia

    Returns
    -------
    scalar
    """

    return mark1 - mark2

def distributions_EMD(d1, d2):
    """
    Implements former Utils.emd() by Patricia

    Parameters
    ----------
    d1 : ProbabilityDistribution
    d2 : ProbabilityDistribution

    Returns
    -------
    scalar
            Earth mover's distance

    """
    return ss.wasserstein_distance(d1.get_probs(), d2.get_probs()) / len(d1.get_probs())

def calculate_error(d1, d2):
    """
    Implements former Utils.calculateError() by Patricia

    Returns
    -------
    scalar
            Earth mover's distance

    """
    square_sum = 0.0
    for i in range(len(d1)):
        square_sum += pow( distributions_EMD(d1[i], d2[i]), 2)
    return math.sqrt(square_sum)

def calculate_distance(mark1, mark2):
    """
    Implements former Utils.calculateDistance() by Patricia

    Parameters
    ----------
    mark1 : double
    mark2 : double

    Returns
    -------
    double

    """
    return abs(mark1 - mark2) / MAX_MARK_VALUE

def argmin_EMD(d1, d2):
    """
    Implements former Utils.min_emdOperator() by Patricia

    Returns
    -------
    ProbabilityDistribution
                            Implementation of eq. 5

    """
    global ideal_difference

    if (distributions_EMD(d1, get_ideal_difference_distribution()) <=
        distributions_EMD(d2, get_ideal_difference_distribution())):
        return d1
    return d2


def to_file(string, folder_name, file_name, ext):
    """
    Implements former Utils.writeToFile() by Patricia

    Parameters
    ----------
    string : str
    folder : str
    file_name : str
    experiment_number : int
    ext : str

    """
    if (not folder_name.is_dir()):
        folder_name.mkdir(parents=True, exist_ok=True)
    file = folder_name / f'{file_name}.{ext}'
    file.touch()

    with open(file, 'a') as f:
        f.write(string)


def compute_algorithm_error(submissions, teacher, automatic_assessments): 
    """
    Implements former SynthDataExperiment_Ranking.calculateError() by Patricia

    Parameters
    ----------
    teacher: Peer

    automatic_assessments: list
        A list of assessments

    Returns
    -------
    double
    """


    error = 0.0
    count = 0

    for submission in submissions:
        found = False

        for assessment in automatic_assessments:
            if (assessment.get_submission().__eq__(submission)):
                found = True
                error += calculate_distance(teacher.get_mark(submission), assessment.get_mark())
                count += 1
                break

        if (not found): # There is no deduced assessment, we asume a default mark
            default_mark = MAX_MARK_VALUE / 2
            error += calculate_distance(teacher.get_mark(submission), default_mark)
            count += 1

    if (count > 0):
        error /= count
    return error

def save_status(service_eigentrust, model_type):
    """
    Saves the current state of the algorithm, including all the computed 
    probability distributions

    Parameters
    ----------
    service_eigentrust: ServiceEigentrust

    model_type: str
        'random' or 'ranking'
    """

    if model_type.lower() == 'random':
        pickle_object = FM().random_model    
    elif model_type.lower() == 'ranking_paas':
        pickle_object = FM().ranking_paas
    elif model_type.lower() == 'ranking_mie':
        pickle_object = FM().ranking_mie
    else: 
        raise ValueError('The model name is not correct')
    
    with pickle_object.open('wb') as model:
        pickle.dump(service_eigentrust, model)

def load_status(model_type):
    """
    Loads the last state of the algorithm, recovering all the computed
    probability distributions

    Parameters
    ----------
    model_type: str
        'random' or 'ranking'

    Returns
    -------
    service_eigentrust: ServiceEigentrust
    """

    if model_type.lower() == 'random':
        pickle_object = FM().random_model    
    elif model_type.lower() == 'ranking_paas':
        pickle_object = FM().ranking_paas
    elif model_type.lower() == 'ranking_mie':
        pickle_object = FM().ranking_mie
    else: 
        raise ValueError('The model name is not correct')

    with pickle_object.open('rb') as model: 
        return pickle.load(model)

def load_synthetic_data():
    """
    Loads the synthetic data used in previous run

    Returns
    -------
    dict
    """

    pickle_object = FM().data_file    

    with pickle_object.open('rb') as data_file: 
        return pickle.load(data_file)


def save_synthetic_data(data):

    """
    Saves the current state of the algorithm, including all the computed 
    probability distributions

    Parameters
    ----------
    data: dict

    """
    
    pickle_object = FM().data_file
    with pickle_object.open('wb') as data_file:
        pickle.dump(data, data_file)



