import sys
import os
import pickle
import multiprocessing as mp
import functools
import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--filepaths', action='store', type=str, required=True, default='features_out.csv')
parser.add_argument('--overwrite', action='store', type=str, required=False, default='no')
parser.add_argument('--threads', action='store', type=int, required=False, default=12)
args = parser.parse_args()
overwrite = args.overwrite== 'yes'

df = pd.read_csv(sys.argv[1])

def safety(filepath):
    with open(filepath, mode='rb') as f:
        process = pickle.load(f)

    involved = pd.Series(process.state.move_log)
    out_path = os.path.join(os.path.split(filepath)[-1], "move_log.csv")
    pd.DataFrame(involved).to_csv(out_path)

with mp.Pool(processes=args.threads) as pool:
    pool.map(safety, list(df.filepath))
