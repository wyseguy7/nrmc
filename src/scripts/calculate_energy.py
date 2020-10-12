import sys
import os
import pickle
import json
import multiprocessing as mp

import pandas as pd

sys.path.append('/gtmp/etw16/nonreversiblecodebase/')
from src.nrmc.analytics import coreset_transitions

df = pd.read_csv(sys.argv[1])

def population_balance_full(state):
    return

def compactness_full(state):
    pass


score_func_to_calc = {

    'cut_length': lambda state: len(state.contested_edges),
    'compactness': compactness_full,
    'population_balance': population_balance_full

}



def compute_energy(process):
    log_score = 0
    for (score_weight, _), score_func in zip(process.score_list, process.score_funcs) :
        log_score += score_weight*score_func(process.state)
    return log_score




def proc_data(data):
    transitions = [(0, data[0])]
    state = data[0]
    for i in range(len(data)):
        if data[i] != -1 and data[i] != state:
            transitions.append((i, data[i]))
            state = data[i]

    return transitions

def write_transitions(filepath, overwrite=False):
    folder, filename = os.path.split(filepath)
    out_path = os.path.join(folder, 'transitions.json')
    if os.path.exists(out_path) and not overwrite:
        return


    with open(filepath, mode='rb') as f:
        process = pickle.load(f)

    transitions = [(i, -1)[int(i is None)] for i in coreset_transitions(process, interval=0.5)]
    trans = proc_data(transitions)
    with open(out_path, mode='w') as f:
        json.dump(trans, f)

with mp.Pool(processes=4) as pool: # how many should we use here?
    pool.map(write_transitions, list(df.filepath))
