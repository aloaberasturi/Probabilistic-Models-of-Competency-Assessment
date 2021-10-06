from experiments import run_experiment

"""
Runs experiment with synthetic data. 
Parameters
----------

model_name ('bivariate' / 'pg1') : str
    The name of the model. 

criterion ('std' / 'random') : str
    Criterion followed to choose next ground truth to observe. 

error_type ('pct' / 'rmse') : str
    Error function
"""
run_experiment(model_name="bivariate", criterion='std', error_type='rmse') # arguments: "bivariate" / "pg1" and 'std' or 'random'