import os
import pickle
import multiprocessing as mp
import functools
import argparse

import pandas as pd

from nrmc.analytics import compute_marginals


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

def write_marginals(filepath, vote_map=None, overwrite=False):

    folder_path = os.path.split(filepath)[0]
    out_path = os.path.join(folder_path, 'marginals.csv')

    if os.path.exists(out_path) and not overwrite:
        return

    with open(filepath, mode='rb') as f:
        process = pickle.load(f)

    resp = compute_marginals(process, vote_map)
    df = pd.DataFrame(resp)
    df.to_csv(out_path, index=None)

func = functools.partial(write_marginals, overwrite=overwrite, vote_map=vote_map)
df = pd.read_csv(args.filepaths)


def safety(filepath):
    try:
        return func(filepath)
    except Exception as e:
        print(e)

with mp.Pool(processes=args.threads) as pool:
    pool.map(safety, list(df.filepath))