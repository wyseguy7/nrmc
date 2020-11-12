import sys
import os
import pickle
import json
import multiprocessing as mp

import pandas as pd

sys.path.append('/gtmp/etw16/nonreversiblecodebase/')

df = pd.read_csv(sys.argv[1])





def calculate_energy(process, energy=0):

    energy_scores = []
    for move in process.state.move_log:

        process._initial_state.handle_move(move) # handle it


        for updater in process.score_updaters:
            updater(process.state)

        if move is not None:
            node_id, old_color, new_color = move
            energy += process.score_proposal(node_id, old_color, new_color, process.state)
        energy_scores.append(energy)

    return energy_scores



def write_energies(filepath, overwrite=False):
    folder, filename = os.path.split(filepath)
    out_path = os.path.join(folder, 'transitions.json')
    if os.path.exists(out_path) and not overwrite:
        return


    with open(filepath, mode='rb') as f:
        process = pickle.load(f)
        energy_scores = calculate_energy(process)
        df = pd.DataFrame(energy_scores)
        df.columns = ['energy']
        df.to_csv(os.path.join(folder, "energy.csv"), index=None)


with mp.Pool(processes=4) as pool: # how many should we use here?
    pool.map(write_energies, list(df.filepath))
