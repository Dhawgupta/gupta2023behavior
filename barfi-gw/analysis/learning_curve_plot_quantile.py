'''
This code will produce the learning curve for different agents
that are specified in the json files
Status : Complete (not completed the key based best parameter selection part)
'''
import os, sys, time, copy
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import numpy as np

from src.utils.json_handling import get_sorted_dict
from analysis.utils import find_best, smoothen_runs
from src.utils.formatting import create_folder
from analysis.colors import agent_colors, lines_type_attribute


# read the arguments etc
if len(sys.argv) < 2:
    print("usage : python analysis/learning_curve.py legend(y/n) <list of json files>")
    exit()

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12 
BIGGEST_SIZE = 25 

# plt.rc('font', size=BIGGER_SIZE )          # controls default text sizes
# plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=BIGGEST_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=BIGGEST_SIZE)    # fontsize of the tick labels
# plt.rc('xtick', titlesize=BIGGEST_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', titlesize=BIGGEST_SIZE)
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# assert sys.argv[1].lower() in ['y' ,'n'] , "[ERROR], Choose between y/n"
# show_legend = sys.argv[1].lower() == 'y'
json_files = sys.argv[1:] # all the json files

json_handles = [get_sorted_dict(j) for j in json_files]




def  plot(ax , data, label = None , color = None):
    # mean =  data['mean'].reshape(-1)
    median = data['quantiles'][1].reshape(-1)
    quantile_lower = data['quantiles'][0].reshape(-1)
    quantile_upper = data['quantiles'][2].reshape(-1)
    median = smoothen_runs(median)
    quantile_upper = smoothen_runs(quantile_upper)
    quantile_lower = smoothen_runs(quantile_lower)
    # mean = smoothen_runs(mean)
    # mean = np.nan_to_num(mean)
    # stderr = data['stderr'].reshape(-1)
    if color is not None:
        base, = ax.plot(median, label = label, linewidth = 3, color = color)
    else:
        base, = ax.plot(median, label=label, linewidth=2)
    ax.fill_between(range(median.shape[0]), quantile_lower, quantile_upper, color = base.get_color(),  alpha = 0.2     )

def plot_multi_key(ax, datas, keys, label = None, color = None):
    for k in keys:
        data = datas[k]
        # mean =  data['mean'].reshape(-1)
        # mean = np.nan_to_num(mean)
        # mean = smoothen_runs(mean)
        median = data['quantiles'][1].reshape(-1)
        median = smoothen_runs(median)
        quantile_lower = data['quantiles'][0].reshape(-1)
        quantile_upper = data['quantiles'][2].reshape(-1)
        quantile_upper = smoothen_runs(quantile_upper)
        quantile_lower = smoothen_runs(quantile_lower)
        # stderr = data['stderr'].reshape(-1)
        if color is not None:
            base, = ax.plot(median, label = label, linewidth = 3, color = color, linestyle = lines_type_attribute[k] )
        else:
            base, = ax.plot(median, label=label, linewidth=5)
        # (low_ci, high_ci) = confidence_interval(mean, stderr)
        # print(median)
        # print(quantile_lower)
        # print(quantile_upper)
        ax.fill_between(range(median.shape[0]), quantile_lower, quantile_upper, color = base.get_color(),  alpha = 0.1     )
        # ax.fill_between(range(median.shape[0]), quantile_upper, quantile_lower, color = base.get_color(),  alpha = 0.1     )

'''
This when you want two scales on the opposite ends of the y axis otherwise use the multi_key code'''
def plot_two_key(ax, ax2,  datas, keys, label = None, color = None):
    k1, k2 = keys
    data = datas[k1]
    # mean =  data['mean'].reshape(-1)
    # mean = np.nan_to_num(mean)
    # # print(label, mean)
    # mean = smoothen_runs(mean)
    # stderr = data['stderr'].reshape(-1)
    median = data['quantiles'][1].reshape(-1)
    quantile_lower = data['quantiles'][0].reshape(-1)
    quantile_upper = data['quantiles'][2].reshape(-1)
    median = smoothen_runs(median)
    quantile_upper = smoothen_runs(quantile_upper)
    quantile_lower = smoothen_runs(quantile_lower)

    if color is not None:
        base, = ax.plot(median, label = label, linewidth = 3, color = color, linestyle = lines_type_attribute[k1] )
    else:
        base, = ax.plot(median, label=label, linewidth=2)
    
    # (low_ci, high_ci) = confidence_interval(mean, stderr)
    ax.fill_between(range(median.shape[0]), quantile_lower, quantile_upper, color = base.get_color(),  alpha = 0.1     )
    ax.set_yscale('log')
    
    if ax2 is None:
        ax2 = ax.twinx()
    data = datas[k2]
    # mean =  data['mean'].reshape(-1)
    # mean = np.nan_to_num(mean)
    # mean = smoothen_runs(mean)
    # stderr = data['stderr'].reshape(-1)
    median = data['quantiles'][1].reshape(-1)
    median = smoothen_runs(median)
    quantile_lower = data['quantiles'][0].reshape(-1)
    quantile_upper = data['quantiles'][2].reshape(-1)
    quantile_upper = smoothen_runs(quantile_upper)
    quantile_lower = smoothen_runs(quantile_lower)  
    if color is not None:
        base2, = ax2.plot(median, label = label, linewidth = 3, color = color, linestyle = lines_type_attribute[k2] )
    else:
        base2, = ax2.plot(median, label=label, linewidth=2)
    # (low_ci, high_ci) = confidence_interval(mean, stderr)
    ax2.fill_between(range(median.shape[0]), quantile_lower, quantile_upper, color = base.get_color(),  alpha = 0.1     )
    
    return ax2

    
# key_to_plot = 'returns' # the key to plot the data
# keys_to_plot = ['returns', 'gamma']
# keys_to_plot = ['returns', 'aux_returns']
keys_to_plot =  ['returns']
# keys_to_plot = ['aux_returns',  'gamma']
# keys_to_plot =  ['aux_returns']
#keys_to_plot =  ['gamma']
## Legends


fig, axs = plt.subplots(1, figsize = (6, 4 ), dpi = 300)
axs2 = None
for en, js in enumerate(json_handles):
    run, param , data = find_best(js, data = 'returns', metric = 'auc')
    # print(param)
    agent = param['agent']
    print(agent)
    if agent == 'REINFORCE':
        label = 'Baseline'
    elif agent  == 'REINFORCEPotential':
        label = 'Baseline + Potential'
    else:
        label = agent
    print(label)
    # plot(axs, data = data[key_to_plot], label = f"{agent}", color = agent_colors[agent] )
    if len(keys_to_plot) == 2:
        axs2 = plot_two_key(axs, axs2, data, keys_to_plot, label = f"{label}", color = agent_colors[agent] )
    else:
        plot_multi_key(axs, data, keys_to_plot, label = f"{label}", color = agent_colors[agent] )

# axs.set_ylim([0, 120])
axs.spines['top'].set_visible(False)
figLegend1 = plt.figure(figsize=(8, 0.5))
l1 = plt.figlegend(*axs.get_legend_handles_labels(), loc='upper center', fancybox=True, shadow=True, ncol=6)
for line in l1.get_lines():
    line.set_linewidth(10.0)

# figLegend1.s

axs.spines['right'].set_visible(False)
# axs.set_ylim([-10,120])
axs.set_ylabel('Return', fontsize = 15)
axs.set_xlabel('Episodes', fontsize= 15)
axs.set_title("No Aux Reward", fontsize = 20)
# axs.set_label
axs.tick_params(axis='both', which='major', labelsize=15)
axs.tick_params(axis='both', which='minor', labelsize=15)

if axs2 is not None:
    axs2.set_ylabel('$\gamma', fontsize = 15)
    axs2.tick_params(axis='both', which='major', labelsize=15)
    axs2.tick_params(axis='both', which='minor', labelsize=15)

    axs2.set_ylabel('$\gamma$', fontsize = 15)

    axs2.set_rasterized(True)

axs.set_rasterized(True)
fig.tight_layout()

foldername = './plots'
create_folder(foldername)
# plt.legend()
# get_experiment_name = input("Give the input for experiment name: ")
get_experiment_name = str(input("Enter Name : > "))
fig.savefig(f'{foldername}/learning_curve_plot_{get_experiment_name}.pdf', dpi = 300)
fig.savefig(f'{foldername}/learning_curve_plot_{get_experiment_name}.png', dpi = 300)
figLegend1.savefig(f'{foldername}/legend_plot_{get_experiment_name}.png', dpi = 300)
figLegend1.savefig(f'{foldername}/legend_plot_{get_experiment_name}.pdf', dpi = 300)
