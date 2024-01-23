'''
This code will just printing the best performing parameter for teh given experiments
'''
import os, sys, time, copy
sys.path.append(os.getcwd())
from src.utils.json_handling import get_sorted_dict
from analysis.utils import find_best
from src.utils.formatting import pretty_print_experiment

# read the arguments etc
if len(sys.argv) < 2:
    print("usage : python analysis/learning_curve.py legend(y/n) <list of json files>")
    exit()

metric = sys.argv[1].lower()
assert metric in ['auc', 'last'], "[ERROR] wrong choice"
opt = sys.argv[2].lower()
assert opt in ['train', 'valid'], "[ERROR] wrong choice"
json_files = sys.argv[3:] # all the json files

json_handles = [get_sorted_dict(j) for j in json_files]

for en, js in enumerate(json_handles):
    run, param , data = find_best(js, data = opt , metric = metric)
    print(f"File : {json_files[en]}")
    pretty_print_experiment(param)


