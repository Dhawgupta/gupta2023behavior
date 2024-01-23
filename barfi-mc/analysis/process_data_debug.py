'''
This file will take in json files and process the data across different runs to store the summary
'''
import os, sys, time, copy
sys.path.append(os.getcwd())
import numpy as np

from src.utils.json_handling import get_sorted_dict , get_param_iterable_runs
from src.utils.formatting import create_file_name, create_folder
from analysis.utils import load_different_runs_debug, pkl_saver, pkl_loader

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
    return mean , stderr



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
                returns, gammas, aux_returns, entropies, gradients = load_different_runs_debug(i)
                mean_return, stderr_return = process_runs(returns)
                # train
                returns_data = {
                    'mean' : mean_return,
                    'stderr' : stderr_return
                }
                # gamma
                mean_gammas, stderr_gammas = process_runs(gammas)
                # train
                gammas_data = {
                    'mean': mean_gammas,
                    'stderr': stderr_gammas
                }

                mean_aux, stderr_aux = process_runs(aux_returns)
                # train
                aux_data = {
                    'mean': mean_aux,
                    'stderr': stderr_aux
                }

                mean_entropy, stderr_entropy = process_runs(entropies)
                # train
                entropy_data = {
                    'mean': mean_entropy,
                    'stderr': stderr_entropy
                }

                mean_gradient, stderr_gradient = process_runs(gradients)
                gradient_data = {
                    'mean' : mean_gradient,
                    'stderr' : stderr_gradient
                }
                pkl_saver({
                        'returns' : returns_data,
                        'gamma' : gammas_data,
                        'aux_returns' : aux_data,
                        'entropies' : entropy_data,
                        'gradients' : gradient_data
                    }, filename)


    # print(iterables)

if __name__ == '__main__':
    process_data_interface(json_handles)