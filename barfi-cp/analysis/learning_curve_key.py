'''
This file will be used to plot learning curves for 1 key
example :
`python analysis/learning_curve_key_key.py model_partition model_std experiments/debug.json
Plots each json file on a separete graph.

'''
import os, sys, time, copy
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import numpy as np

from src.utils.json_handling import get_sorted_dict, get_sorted_dict_loaded
from analysis.utils import find_best_key, smoothen_runs, find_best
from src.utils.formatting import get_folder_name, create_folder
from analysis.colors import  agent_colors

# read the arguments etc
if len(sys.argv) < 2:
    print("usage : python analysis/learning_curve_key_key.py json_file")
    exit()

key1 = sys.argv[1]
json_files = sys.argv[2:]
json_handles = [get_sorted_dict(j) for j in json_files]
# key1 = 'eval_greedy'
metric = 'auc'
opt = 'returns'
show_legend = True



def confidence_interval(mean, stderr):
    return (mean - stderr, mean + stderr)


def  plot(ax , data, label = None , color = None, line_style = None):
    mean =  data['mean'].reshape(-1)
    mean = smoothen_runs(mean)
    stderr =  data['stderr'].reshape(-1)
    if color is None:
        base, = ax.plot(mean, label=label, linewidth=2, linestyle=line_style)
    else:
        base, = ax.plot(mean, label = label, linewidth = 2, color = color, linestyle = line_style)
    # base, = ax.plot(mean, label = label, linewidth = 2, color = color, linestyle = line_style)
    (low_ci, high_ci) = confidence_interval(mean, stderr)
    ax.fill_between(range(mean.shape[0]), low_ci, high_ci, color = base.get_color(),  alpha = 0.2  )

key_to_plot = 'returns' # the key to plot the data

fig, axs = plt.subplots(1, figsize=(6, 4), dpi=300)
axs = [axs]
for js in json_handles:
    # fig, axs = plt.subplots(1, figsize=(6, 4), dpi=300)
    # axs = [axs]
    agent = js['agent']
    # print(js['agent'])
    problem = js['env_name']
   
    if agent == 'something':
        # runs, params, keys, data = find_best_key(js, data = opt , key = ['pretrain'], metric = metric)
        pass
    else:
        runs, params, keys, data = find_best_key(js, key = [key1], data = 'returns')

    print(keys)
    for k in keys: # pretrain : true, fals
        for i, key in enumerate(['returns']):
            if agent != 'backprop':
                label = None
                label = f'{k[0]}'

            line_style = None
            plot(axs[i], data = data[k][key], label = label,  line_style= line_style )
            axs[i].spines['top'].set_visible(False)
            if show_legend:
                axs[i].set_title(f'{key} loss')
                axs[i].legend()

            axs[i].spines['right'].set_visible(False)
            axs[i].tick_params(axis='both', which='major', labelsize=8)
            axs[i].tick_params(axis='both',  which='minor', labelsize=8)
            axs[i].set_rasterized(True)

fig.tight_layout()

foldername = './plots'
create_folder(foldername)
# plt.legend()
# get_experiment_name = input("Give the input for experiment name: ")
# layers = js['model_specification'][0]['num_layers']

plt.savefig(f'{foldername}/learning_curve_{key1}_{agent}-{opt}-{metric}-{problem}.pdf', dpi=300)
plt.savefig(f'{foldername}/learning_curve_{key1}_{agent}-{opt}-{metric}-{problem}.png', dpi=300)