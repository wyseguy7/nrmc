import sys
import os
import pickle
import json
import multiprocessing as mp

import pandas as pd

sys.path.append('/gtmp/etw16/nonreversiblecodebase/')
from src import coreset_transitions

df = pd.read_csv(sys.argv[1])

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
