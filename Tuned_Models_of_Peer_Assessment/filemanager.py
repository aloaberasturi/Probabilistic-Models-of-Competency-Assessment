from datetime import datetime
import os

class FileManager():

    def __init__(self):

        date = datetime.now() # timeprint to disinguish experiments
        self.experiment_folder = f'./experiment'#_{date}'
        if not os.path.exists(self.experiment_folder):
            os.makedirs(self.experiment_folder)

    def create_paths(self, S):

        date = datetime.now() # timeprint to distinguish runs of an experiment
        rmse_file = f'{self.experiment_folder}/rmse.csv'
        rmse_plot = f'{self.experiment_folder}/rmse.jpg'
        run_folder = f'{self.experiment_folder}/main'#{date}
        model_pkl = f'{self.experiment_folder}/main/model.pkl'
        fit_pkl = f'{self.experiment_folder}/main/fit.pkl'
        histograms_folder = f'{run_folder}/histograms'
        samples_s_histogram = f'{histograms_folder}/{S}_samples_s.jpg'
        samples_z_histogram = f'{histograms_folder}/{S}_samples_z.jpg'
        graders_histogram = f'{histograms_folder}/{S}_graders-gradees.jpg'


        paths = {
            'folder': run_folder,
            'model pkl': model_pkl,
            'fit pkl': fit_pkl,
            'histograms folder': histograms_folder,
            'rmse file': rmse_file,
            'rmse plot': rmse_plot,
            'samples s histogram': samples_s_histogram,
            'samples z histogram': samples_z_histogram,
            'graders histogram': graders_histogram
        }

        # Create folders for each run
        if not os.path.exists(paths['folder']):
            os.makedirs(paths['histograms folder']) 

        return paths
