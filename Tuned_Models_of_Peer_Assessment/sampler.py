import pandas as pd
import matplotlib.pyplot as plt
import common
import numpy
import pystan
import pickle


class Sampler():
    """
    A class containing all methods related to bayesian parameter estimation
    """
    def __init__(self, data_points, model, stan_mode):
        self.stan_input = data_points
        self.stan_mode = stan_mode
        self.model = model


    def sampling(self):
        """
        Main function. Calls stanmodel.sampling(), extracts its contents, generates histograms of posterior predictive
        samples and saves the information into .csv files.
        """

        # 1) Open compiled model or compile it
        try:        
            posterior = pickle.load(open(common.paths["model pkl"], 'rb'))
        except FileNotFoundError:
            posterior = pystan.StanModel(model_code=self.model)  
            with open(common.paths["model pkl"], 'wb') as f:
                pickle.dump(posterior, f)

        # 2) Sample
        if self.stan_mode == 'sampling':
            fit = posterior.sampling(data=self.stan_input, control=dict(max_treedepth=15))        
        elif self.stan_mode == 'optimizing':
            fit = posterior.optimizing(data=self.stan_input, iter=90000) # histograms are plotted using MAP estimates

        return fit
    