from mimetypes import init
import os, sys, time
sys.path.append(os.getcwd())
import torch
import numpy as np
import logging
import warnings
from collections import namedtuple
import matplotlib.pyplot as plt

# based on   's codebase
import argparse
from datetime import datetime
from src.config import Config

from src.solver import Solver
import torch


# from src.algorithms.registry import get_agent
# from src.environments.registry import get_environment
from src.utils.json_handling import get_sorted_dict, get_param_iterable
from src.utils.data_structures import transition
from analysis.utils import pkl_saver
from src.utils.formatting import create_file_name

from analysis.utils import smoothen_runs
warnings.filterwarnings("ignore", category=UserWarning)




def get_output_filename(ex):
    folder, filename = create_file_name(ex)
    if not os.path.exists(folder):
        time.sleep(2)
        try:
            os.makedirs(folder)
        except:
            pass
    output_file_name = folder + filename
    return output_file_name

def init_dict(exp):
    # fill contents according to   's codebase from the run maze
    exp['debug'] = True
    exp['save_model'] = True
    exp['log_output'] = '' # no output logging
    exp['log_output'] = 'term_file'
    exp['save_count'] = 100
    exp['experiment'] = 'Data'
    exp['folder_suffix'] = 'Default'
    exp['restore'] = False
    exp['gpu'] = False
    return exp

def main(mode = 'train'):
    # load different stuff
    t = time.time()
    if len(sys.argv) < 3:
        print("usage : python src/main.py json_file idx")
        exit()
    json_file = sys.argv[1]
    idx = int(sys.argv[2])
    d = get_sorted_dict(json_file)
    experiments = get_param_iterable(d)
    experiment = experiments[idx % len(experiments)]
    # print(experiment)
    folder, filename = create_file_name(experiment)
    experiment = init_dict(experiment)
    torch.set_num_threads(1)
    # exp_list = convert_dict_to_list(experiment)
    # experiment_obj = parse.parse_args(exp_list)

    experiment_obj = namedtuple("experiment", experiment.keys())(*experiment.values())
    # experiment_obj.__dict__ = experiment
    # set the seeds
    config = Config(experiment_obj, experiment)
    solver = Solver(config=config)



    if not os.path.exists(folder):
        time.sleep(2)
        try:
            os.makedirs(folder)
        except:
            pass

    output_file_name = folder + filename
    if os.path.exists(output_file_name + '.dw'):
        print("Run Already Complete - Ending Run")
        exit()

    if mode == 'train':
        ret, gam, aux, data = solver.train(max_episodes=config.max_episodes)

    elif mode == 'eval':
        solver.eval(max_episodes=int(1e5))

    elif mode == 'collectdata':
        solver.collect(max_episodes=int(2e5))

    else:
        return ValueError

    x1 = data['x1']
    x3 = data['x3']

    entropies = data['entropies']
    gradients = data['gradients']

    gradient_arrays = []
    for x in gradients[0]:
        gradient_arrays.append([])
    for i, x in enumerate(gradients):
        for j, c in enumerate(x):
            if len(c) == 0:
                gradient_arrays[j].append(0)
            else:
                gradient_arrays[j].append(np.sum(c))
    x1 = np.array(x1)
    x3 = np.array(x3)
    entropies = np.array(entropies)
    gradients = np.array(gradient_arrays)


    # ret = np.array(ret)
    # gam = np.array(gam)
    # aux = np.array(aux)

    print("Total time taken: {}".format(time.time() - t))
    pkl_saver({
        'returns' : ret,
        'gamma' : gam,
        'aux_returns' : aux,
        'x1' : x1,
        'x3' : x3,
        'entropies' : entropies,
        'gradients' : gradients
    }, output_file_name + '.dw')

    plot = True
    plot = False
    if plot:
        plt.plot(smoothen_runs(ret), label='return')
        plt.plot(smoothen_runs(gam), label='gamma')
        plt.plot(smoothen_runs(aux), label='aux')
        plt.legend()
        plt.savefig(f'plot.png')
    return True

    # t_start = time.time()
    # logging.basicConfig(level=logging.INFO)
    # # do sample based running
    # # num_samples =  experiment['num-samples']
    # num_episodes = experiment['num-episodes']
    #
    # # run experiments
    # # def run_experiment_samples(agent, problem, repr, no_samples, gamma = 0.999):
    # returns = run_experiment_episodes(agent, problem, repr, num_episodes, writer, gamma)
    # pkl_saver({
    #     'returns': returns
    # }, output_file_name + '.dw')
    # end_time = time.time()
    # print(f"Time Taken : {end_time - t_start}")


if __name__ == '__main__':
    main(mode = 'train')