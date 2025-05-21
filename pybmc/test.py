
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(".."))  # adjust if your notebook is deeper
from pybmc.bmc import BayesianModelCombination
from pybmc.inference_utils import USVt_hat_extraction

data = pd.DataFrame({
    'model1': [10.0, 20.0, 30.0],
    'model2': [11.0, 21.0, 31.0],
    'model3': [9.0, 19.0, 29.0],
    'truth':  [10.5, 20.5, 30.5],
    'N': [1, 2, 3],
    'Z': [10, 10, 10]
})

models = ['model1', 'model2', 'model3', 'truth']
bmc = BayesianModelCombination(models, data)

bmc.orthogonalize(data, components_kept=2)

training_options = {
    'iterations': 1000,
    'b_mean_prior': np.zeros(2),
    'b_mean_cov': np.diag(bmc.S_hat**2),
    'nu0_chosen': 1.0,
    'sigma20_chosen': 0.02
}

print("S_hat:", bmc.S_hat)
print("Training options:", training_options)
print("Centered experiment train:", bmc.centered_experiment_train)
print("U_hat:", bmc.U_hat)
print("Vt_hat:", bmc.Vt_hat)

bmc.train(training_options)

print("Samples shape:", bmc.samples.shape)



# In[ ]:




