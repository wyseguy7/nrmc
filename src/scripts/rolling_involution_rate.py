import sys
import os
import pickle
import multiprocessing as mp
import functools
import argparse
from collections import defaultdict
import json

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--filepaths', action='store', type=str, required=True, default='features_out.csv')
parser.add_argument('--overwrite', action='store', type=str, required=False, default='no')
parser.add_argument('--threads', action='store', type=int, required=False, default=12)
parser.add_argument('--truncate', action='store', type=int, required=False, default=0)
args = parser.parse_args()
overwrite = args.overwrite== 'yes'

def write_wait_times(filepath, overwrite=False, truncate=0):
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

    counter = 0
    wait_counter = defaultdict(int)
    for move in process.state.move_log:
        counter += 1

        if move is None:
            wait_counter[counter] += 1
            counter = 0

    with open(out_path, mode='w') as f:
        json.dump(wait_counter, f)


func = functools.partial(write_wait_times, overwrite=overwrite, truncate=args.truncate)
df = pd.read_csv(args.filepaths)
print(len(df))
def safety(filepath):
    try:
        return func(filepath)
    except Exception as e:
        print(e)

with mp.Pool(processes=args.threads) as pool:
    pool.map(safety, list(df.filepath))
