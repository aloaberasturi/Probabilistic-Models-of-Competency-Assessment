import itertools
import random
from numpy.lib.financial import npv
from numpy.random import normal as normal_distr
from numpy.random import uniform
from numpy.random import gamma as gamma_distr
from numpy.random import multivariate_normal
from scipy.stats import invgamma as inv_gamma_distr
from scipy.stats import truncnorm
import numpy as np
import pandas as pd

paths = None

def truncnorm_sample(min_v, max_v, loc, scale, size):
    a, b = (min_v - loc) / scale, (max_v - loc) / scale
    return truncnorm.rvs(a, b, loc=loc, scale=scale, size=size)

def compute_matrix(name, S, uu, vv, z, mu_matrix = None):
    if name == 'mu':
        count_matrix = np.zeros((S, S))
        mu_matrix    = np.zeros((S, S))    
    
    elif name == 'sigma':
        variance_matrix = np.zeros((S, S))    

    for i in range(1, S+1):
        indices = np.where(uu == i) # Stan indexing
        graders_pairs = list(itertools.combinations(list(vv[indices]), 2))
        z_pairs = list((itertools.combinations(list(z[indices]), 2)))
        
        for p_ii, z_p in zip(graders_pairs, z_pairs):
            g1 = p_ii[0] - 1 
            g2 = p_ii[1] - 1
            z1 = z_p[0]
            z2 = z_p[1]

            if name == 'mu':
                mu_matrix[g1][g2] += z1-z2
                mu_matrix[g2][g1] = -mu_matrix[g1][g2]
                count_matrix[g1][g2] += 1.0 # Stan indexing
                count_matrix[g2][g1] = count_matrix[g1][g2] 

            elif name == 'sigma':
                variance_matrix[g1][g2] += (z1-z2 - mu_matrix[g1][g2])**2
                variance_matrix[g2][g1] = variance_matrix[g1][g2] 
    

    if name == 'mu':
        mu_matrix = np.divide(mu_matrix, count_matrix)
        return_array = [mu_matrix, count_matrix]
    
    else: return_array = [variance_matrix]

    return return_array


def generate_data_pg1(n, S, hyperparameters, dist = 'uniform'):

    alpha = hyperparameters['alpha']
    beta  = hyperparameters['beta']
    gamma = hyperparameters['gamma']
    eta   = hyperparameters['eta']
    mu    = hyperparameters['mu']

    N = int(S*n)
    
    # define who evaluates whom avoiding self-assessment and repeating pairs grader-gradee
    # the assignment is such that every student is graded by three peers

    combinations = list(itertools.permutations(list(range(1,S+1)),2)) # max num combinations = S^{2}-S

    flatten = lambda t: [item for sublist in t for item in sublist]    
    uv = flatten([random.sample([i for i in combinations if i[0] == k],n) for k in range(1, S+1)])
    
    uu = np.array([a for (a,_) in uv])
    vv = np.array([b for (_,b) in uv])

    # generate sigma
    sigma = np.array([inv_gamma_distr.rvs(alpha, scale=beta)]) 

    # generate b
    vector_b = np.array([normal_distr(0, eta) for _ in range(0,S)]) # one element per student    

    # generate s
    vector_s = np.array([normal_distr(mu, gamma) for _ in range(0,S)])   
    # vector_s = truncnorm_sample(0,3, loc=mu, scale=gamma, size=S).tolist()

    # generate z

    if dist == 'normal':
        z = np.array([normal_distr(vector_s[uu[n]-1] + vector_b[vv[n]-1], sigma) for n in range(0,N)]).squeeze()
    elif dist == 'uniform':
        z = uniform(low = 0, high = 3.01, size = int(N/2))

    for n in range(int(N/2), N):
        z = np.append(z, vector_s[uu[n]-1])
    z = z.tolist()

    return {'S': S, 'N': N, 'z':z, 'uu':uu, 'vv':vv, 'uv':uv, 's': vector_s, 'sigma': sigma, 'b': vector_b }

def generate_data_bivariate(n, S, hyperparameters):

    N = int(n*S) # number of z's
    mu = hyperparameters['mu']
    sigma = hyperparameters['sigma']

    combinations = list(itertools.permutations(list(range(1,S+1)),2)) # max num combinations = S^{2}-S

    flatten = lambda t: [item for sublist in t for item in sublist]    
    uv = flatten([random.sample([i for i in combinations if i[0] == k],n) for k in range(1, S+1)])

    uu = np.array([a for (a,_) in uv])
    vv = np.array([b for (_,b) in uv])

    # generate s
    s = truncnorm_sample(0, 3, loc=1.5, scale=0.7, size=S) 
    # s = np.array([normal_distr(1.5, 1.1) for _ in range(0,S)])   

    graders_pairs = [] # get all pairs of graders with some evaluation in common

    for i in range(1, S+1):
        indices = np.where(uu == i) # Stan indexing
        graders_pairs.extend(list(itertools.combinations(list(vv[indices]), 2)))

    ragged_z = [] # pairs of peer grades following bivariate distributions 
    lengths = [] # instead of creating matrices of varying number of columns, I follow this guidelines: https://mc-stan.org/docs/2_26/stan-users-guide/ragged-data-structs-section.html
    z = [None] * N
    for p in graders_pairs: 
        p = list(p)
        count = 0
        ii_1 = np.where(vv == p[0])[0]
        ii_2 = np.where(vv == p[1])[0]
        _, x_ind, y_ind = np.intersect1d(uu[ii_1], uu[ii_2], return_indices=True)
        for i,j in zip(ii_1[x_ind.tolist()], ii_2[y_ind.tolist()]):
            mu_bi = np.array([mu[vv[i]-1], mu[vv[j]-1]]) 
            sigma_12 = 1.8*np.random.uniform(-1,1) 
            L = np.matrix( 
                [[sigma[vv[i]-1], 0],                                
                 [sigma_12, sigma[vv[j]-1]]
                ]
            )
            sigma_matrix = L * L.transpose() 
            [z[i],z[j]] = multivariate_normal(mu_bi, sigma_matrix) 

            ragged_z.append([z[i],z[j]])
            count += 1
        lengths.append(count)

    gg1 = [int(g1) for (g1,_) in graders_pairs] # index for first grader of each pair of graders with assessments in common
    gg2 = [int(g2) for (_,g2) in graders_pairs] # index for second grader of each pair of graders with assessments in common
    
    data = {
        'z': z,
        'S': S,
        'N': N, 
        'N_pairs': len(graders_pairs),
        'gg1' : gg1,
        'gg2' : gg2,
        'uu': uu.tolist(), 
        'vv': vv.tolist(),
        'N_ragged_z' : len(ragged_z),
        'ragged_z': ragged_z,
        'lengths': lengths,
        's': s.tolist(), 
        'mu': mu, 
        'sigma': sigma,
        'pairs': uv
    }

    return data

def compute_error(fit, ii_miss_p, true_grades, stan_mode, error_type):
    if stan_mode == 'sampling':
        s_tilde = fit['s'].mean(axis=0)
    elif stan_mode == 'optimizing':
        s_tilde = fit['s']

    ii_miss_p = [i-1 for i in ii_miss_p]
    s_tilde_missing = np.take(s_tilde, ii_miss_p) 
    s_missing = np.take(true_grades, ii_miss_p)

    if error_type == 'pct':
        err = (np.absolute(np.rint(s_tilde) - np.rint(true_grades))).mean()
        err_wo_known_parameters = (np.absolute(np.rint(s_tilde_missing - np.rint(s_missing)))).mean()
    elif error_type == 'rmse':
        err = np.sqrt(((s_tilde - true_grades)**2).mean())   
        err_wo_known_parameters = np.sqrt(((s_tilde_missing - s_missing)**2).mean()) 

    rmse_df = pd.DataFrame(np.array([[err, err_wo_known_parameters]]) ,columns=[f'err ({error_type})', 'err w/o known true grades'])
    rmse_file = paths['rmse file']

    with open(rmse_file, 'a') as f:
        rmse_df.to_csv(f, header=f.tell()==0, index=False)
    print(f'error computation added to {rmse_file}')

    return rmse_df

