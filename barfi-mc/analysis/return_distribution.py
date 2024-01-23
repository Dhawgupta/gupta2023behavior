'''
this file assumes you are inputting a single type of param to be plotted
'''
import os, sys, time, copy
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.json_handling import get_sorted_dict
from analysis.utils import find_best, smoothen_runs, load_different_runs
from src.utils.formatting import create_folder
from analysis.colors import agent_colors
import pandas as pd
# read the arguments etc
if len(sys.argv) < 2:
    print("usage : python analysis/learning_curve.py legend(y/n) <list of json files>")
    exit()

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
BIGGEST_SIZE = 25

plt.rc('font', size=BIGGER_SIZE )          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
# plt.rc('xtick', titlesize=BIGGEST_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', titlesize=BIGGEST_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


assert sys.argv[1].lower() in ['y' ,'n'] , "[ERROR], Choose between y/n"
show_legend = sys.argv[1].lower() == 'y'
json_files = sys.argv[2:] # all the json files

json_handles = [get_sorted_dict(j) for j in json_files]

def confidence_interval(mean, stderr):
    return (mean - stderr, mean + stderr)

def  plot(ax , data, label = None , color = None):
    mean =  data['mean'].reshape(-1)
    mean = smoothen_runs(mean)
    stderr = data['stderr'].reshape(-1)
    if color is not None:
        base, = ax.plot(mean, label = label, linewidth = 3, color = color)
    else:
        base, = ax.plot(mean, label=label, linewidth=2)
    (low_ci, high_ci) = confidence_interval(mean, stderr)
    ax.fill_between(range(mean.shape[0]), low_ci, high_ci, color = base.get_color(),  alpha = 0.2     )

key_to_plot = 'returns' # the key to plot the data

fig, axs = plt.subplots(1, figsize = (6, 4 ), dpi = 300)
agent_returns = {}
agent_return_list = []
for en, js in enumerate(json_handles):
    agent = js['agent']
    returns  = load_different_runs(js).mean(axis = 1)
    agent_returns[agent] = returns
    for r in returns:
        agent_return_list.append((agent, r))
all_agents = agent_returns.keys()
# axs.set_ylim([0, 120])
df = pd.DataFrame(data=agent_return_list)
sns.displot(df, x = 1, kind = "kde", fill = False, hue = 0, palette=agent_colors)
plt.xlabel('Returns')
# axs.spines['top'].set_visible(False)
# # plt.show()
# if show_legend:
#     axs.set_title(f'Learning Curve')
#     axs.legend()
#
# axs.spines['right'].set_visible(False)
# axs.set_ylabel('Density')
# axs.set_xlabel('Returns')
# axs.tick_params(axis='both', which='major', labelsize=12)
# axs.tick_params(axis='both', which='minor', labelsize=12)
# axs.set_rasterized(True)
# fig.tight_layout()

foldername = './plots'
create_folder(foldername)
# plt.legend()
# get_experiment_name = input("Give the input for experiment name: ")
get_experiment_name = 'l1'
plt.savefig(f'{foldername}/density_{get_experiment_name}.pdf', dpi = 300)
plt.savefig(f'{foldername}/density_{get_experiment_name}.png', dpi = 300)


