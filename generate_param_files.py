import numpy as np
import itertools
import csv
import os
import glob

# generate less params if it's a test run
real_run = False



## this generates parameter values for the main model pipeline
    
# empty all files first
files = glob.glob('params/param_vals/*')
for f in files:
    os.remove(f)

# parameters
if real_run:
    param_dict = {'mean_mu': np.arange(-1, 1, 0.2).tolist(),
                'mean_sd': np.arange(1, 4, 0.5).tolist(),
                'sd_scale': np.arange(2, 7, 0.5).tolist(),
                'noise_scale': np.arange(0.1, 4, 0.5).tolist()}
else:
    param_dict = {'mean_mu': [-0.3,0.3],
                'mean_sd': [1],
                'sd_scale': [2],
                'noise_scale': [1]}

all_params_combinations = itertools.product(*list(param_dict.values()))

# write param value files
for param_idx, combination in enumerate(list(all_params_combinations)):
    with open('params/param_vals/params' + str(param_idx) + '.csv', 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(list(combination))

# write param name file
with open('params/param_names.csv', 'w') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(param_dict.keys()) 

