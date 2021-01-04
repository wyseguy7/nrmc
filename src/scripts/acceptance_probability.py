import os
import pickle
import multiprocessing as mp
import functools
import argparse
import copy
from collections import defaultdict
import json
import random

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--filepaths', action='store', type=str, required=True, default='features_out.csv')
parser.add_argument('--overwrite', action='store', type=str, required=False, default='no')
parser.add_argument('--threads', action='store', type=int, required=False, default=12)
parser.add_argument('--truncate', action='store', type=int, required=False, default=0)
parser.add_argument('--guess', action='store_true')
args = parser.parse_args()
overwrite = args.overwrite == 'yes'


def write_acceptance_probs(filepath, overwrite=False, truncate=0):
    print(filepath)
    folder, filename = os.path.split(filepath)

    out_path = os.path.join(folder, 'involution_wait_times.json')
    if os.path.exists(out_path) and not overwrite:
        print('skipping')
        return

    with open(filepath, mode='rb') as f:
        process = pickle.load(f)
    print('loaded')

    if truncate != 0:
        process.state.move_log = process.state.move_log[:truncate]

    accept_prob = list()
    proposal_list = list()
    move_log = copy.deepcopy(process.state.move_log)
    process.state = process.state._initial_state # wipe the state
    for move in move_log:
        # run the updaters
        for func in process.score_updaters:
            func(process.state)

        if move is None:
            # we have to guess what the proposal is - sample proposals conditioned on MH rejection
            while True:
                prop, prob = process.proposal(process.state)

                u = random.random() # TODO rng here
                if prob < u:
                    # we rejected
                    ap = prob
                    process.handle_rejection(prop, process.state) # we rejected the proposal - break!
                    proposal_list.append(prop)
                    break
                else:
                    # we accepted this, pick another proposal
                    # we need to reject this one but preserve involution state
                    process.handle_rejection(prop, process.state)
                    process.perform_involution()

        else:
            node_id, old_color, new_color = move
            ap = process.score_to_prob(process.score_proposal(node_id, old_color, new_color, process.state))
            process.handle_acceptance(move, process.state)
            proposal_list.append(move)


        accept_prob.append(ap)

    df = pd.DataFrame({"proposal": proposal_list, "acceptance_prob": accept_prob})
    df.to_csv(out_path, index=None)


func = functools.partial(write_acceptance_probs, overwrite=overwrite, truncate=args.truncate)
df = pd.read_csv(args.filepaths)
print(len(df))


def safety(filepath):
    try:
        return func(filepath)
    except Exception as e:
        print(e)


with mp.Pool(processes=args.threads) as pool:
    pool.map(safety, list(df.filepath))
