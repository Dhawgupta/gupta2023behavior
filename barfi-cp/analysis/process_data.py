'''
This file will take in json files and process the data across different runs to store the summary
'''
import os, sys, time, copy
sys.path.append(os.getcwd())
import numpy as np

from src.utils.json_handling import get_sorted_dict , get_param_iterable_runs
from src.utils.formatting import create_file_name, create_folder
from analysis.utils import load_different_runs, pkl_saver, pkl_loader

# read the arguments etc
if len(sys.argv) < 2:
    print("usage : python analysis/process_data.py <list of json files")
    exit()

json_files = sys.argv[1:] # all the json files

# convert all json files to dict
json_handles = [get_sorted_dict(j) for j in json_files]

def process_runs(runs):
    # get mean and std
    mean = np.mean(runs, axis = 0)
    stderr = np.std(runs , axis = 0) / np.sqrt(runs.shape[0])
    quantiles = np.percentile(runs, [5, 50, 95], axis=0)
    return mean , stderr, quantiles



# currentl doesnt not handle frames
def process_data_interface(json_handles):
    for js in json_handles:
        runs = []
        iterables = get_param_iterable_runs(js)
        for i in iterables:
            folder, file = create_file_name(i, 'processed')
            create_folder(folder) # make the folder before saving the file
            filename = folder + file + '.pcsd'
            # check if file exists
            print(filename)
            if os.path.exists(filename):
                print("Processed")

            else:
                returns, gammas, aux_returns = load_different_runs(i)
                mean_return, stderr_return, quantile_return = process_runs(returns)
                # train
                returns_data = {
                    'mean' : mean_return,
                    'stderr' : stderr_return,
                    'quantile' : quantile_return
                
                }
                # gamma
                mean_gammas, stderr_gammas, quantile_gammas = process_runs(gammas)
                # train
                gammas_data = {
                    'mean': mean_gammas,
                    'stderr': stderr_gammas,
                    'quantile' : quantile_gammas

                }

                mean_aux, stderr_aux, quantile_aux = process_runs(aux_returns)
                # train
                aux_data = {
                    'mean': mean_aux,
                    'stderr': stderr_aux,
                    'quantile' : quantile_aux
                }
                pkl_saver({
                        'returns' : returns_data,
                        'gamma' : gammas_data,
                        'aux_returns' : aux_data
                    }, filename)


    # print(iterables)

if __name__ == '__main__':
    process_data_interface(json_handles)