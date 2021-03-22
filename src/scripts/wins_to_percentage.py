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
import numpy as np

from nrmc.analytics import compute_districts_won

parser = argparse.ArgumentParser()
parser.add_argument('--filepaths', action='store', type=str, required=True, default='features_out.csv')
parser.add_argument('--overwrite', action='store', type=str, required=False, default='no')
parser.add_argument('--threads', action='store', type=int, required=False, default=12)
parser.add_argument('--vote_file', action='store', type=str, required=False,
                    default='data/Mecklenburg/Mecklenburg_P__EL12G_PR.txt')

args = parser.parse_args()
overwrite = args.overwrite == 'yes'

vote_df = pd.read_csv(args.vote_file, sep='\t', header=None)
vote_df.columns = ['node_id', 'p1', 'p2', 'total', 'nan_column']
vote_map = {row.node_id: (row.p1, row.p2) for row in vote_df.itertuples(index=False)}

def write_win_estimate(filepath, overwrite=False, wins_max=5):

    folder_path = os.path.split(filepath)[0]
    out_path = os.path.join(folder_path, 'win_estimate.csv')

    if os.path.exists(out_path) and not overwrite:
        return


    wins_path = os.path.join(folder_path, 'wins.csv')
    df = pd.read_csv(wins_path)

    output = np.zeros(shape=(len(df), wins_max+1))

    counts = np.array([0 for _ in range(wins_max+1)])
    for i in range(len(df)):
        counts[df.iloc[i,0]] += 1
        output[i,:] = counts/(i+1) # this is a copy, yes?

    df_out = pd.DataFrame(output)
    df_out.to_csv(out_path, index=None)

func = functools.partial(write_win_estimate, overwrite=overwrite)
df = pd.read_csv(args.filepaths)


def safety(filepath):
    try:
        return func(filepath)
    except Exception as e:
        print(e)


with mp.Pool(processes=args.threads) as pool:
    pool.map(safety, list(df.filepath))
