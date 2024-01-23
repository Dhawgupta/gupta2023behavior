import os, sys
import numpy as np
from src.utils.formatting import create_file_name , get_folder_name

# def check_experiment_not_done(experiment):
#     '''
#     Returns True if experiment is yet to be done
#     '''
#     folder, file = create_file_name(experiment)
#     file_name_check = folder+file + '.dw'
#     # check ifn th efile exists
#     if not os.path.exists(file_name_check):
#         return True
#     return False
#
# def get_list_pending_experiments(experiments):
#     '''
#     Inputs : <list> of expeirments
#     Returns : Index of pending experiments
#     '''
#     # given a list of expeiments
#     pending_experiments = []
#     for idx, exp in enumerate(experiments):
#         if check_experiment_not_done(exp):
#             pending_experiments.append(idx)
#     return pending_experiments


def check_experiment_not_done(experiment, list_of_done_experiments = None):
    '''
    Returns True if experiment is yet to be done
    '''
    folder, file = create_file_name(experiment)
    # print(file)
    file_name_check = folder + file + '.dw'
    if list_of_done_experiments is not None:
        if file + '.dw' not in list_of_done_experiments:
            return True
        return False
    # check ifn th efile exists
    if not os.path.exists(file_name_check):
        return True
    return False

def get_list_pending_experiments(experiments):
    '''
    Inputs : <list> of expeirments
    Returns : Index of pending experiments
    '''
    # given a list of expeiments
    pending_experiments = []
    experiment_no = len(experiments)
    # get folder name and the experiments in those
    foldername = get_folder_name(experiments[0])
    # load all files

    print(foldername)
    # import time
    list_of_done_experiments = None
    if os.path.exists(foldername):
        list_of_done_experiments = os.listdir(foldername)
    # print(list_of_done_experiments)
    # time.sleep(10)

    for idx, exp in enumerate(experiments):
        print(f'Checking [{idx}/{experiment_no}]\r' , end = "")
        if check_experiment_not_done(exp, list_of_done_experiments):
            pending_experiments.append(idx)
    return pending_experiments


def get_list_pending_experiments_faster(experiments):
    pending_experiments = []
    experiment_num = len(experiments)
    foldername = get_folder_name(experiments[0])
    list_of_done_experiments = None
    if os.path.exists(foldername):
        list_of_done_experiments = os.listdir(foldername)
    file_name_checks = []
    for exp in experiments:
        folder, file = create_file_name(exp)
        file_name_checks.append( file + '.dw')
    # sort the list of all done experiments
    argsort = np.argsort(file_name_checks)
    file_name_checks = np.array(file_name_checks)[argsort]
    counter_done = 0
    counter_exp = 0

    if list_of_done_experiments is  None or len(list_of_done_experiments) == 0:
        list_of_done_experiments = []
        return argsort
        # return list(range(len(experiments)))
    # sort the list of done experiments
    list_of_done_experiments = sorted(list_of_done_experiments)
    # print(file_name_checks[:5])
    # print(list_of_done_experiments[:5])


    while counter_exp < len(file_name_checks):
        print(f'Checking [{counter_exp}/{experiment_num}]\r' , end = "")
        if file_name_checks[counter_exp] == list_of_done_experiments[counter_done]:
            counter_done += 1
            counter_exp += 1
        elif file_name_checks[counter_exp] > list_of_done_experiments[counter_done]:
            counter_done += 1
        else:
            pending_experiments.append(argsort[counter_exp])
            counter_exp += 1
    # for idx,file in enumerate(file_name_checks):
    #     print(f'Checking [{idx}/{experiment_num}]\r' , end = "")
    #     if file == list_of_done_experiments[counter_done]:
    #         counter_done += 1
    #     else:
    #         pending_experiments.append(argsort[idx])
    return pending_experiments
