import sys
import os
import pickle
import multiprocessing as mp
import functools
import argparse

import pandas as pd

sys.path.append('/gtmp/etw16/nonreversiblecodebase/')
from src.analytics import compute_autocorr_new

data_points = ['district_1', 'times_contested', 'node_flips']

parser = argparse.ArgumentParser()
parser.add_argument('--filepaths', action='store', type=str, required=True, default='features_out.csv')
parser.add_argument('--intervals', action='store', type=int, required=False, default=200)
parser.add_argument('--max_distance', action='store', type=int, required=False, default=5000000)
parser.add_argument('--points', action='store', type=int, required=False, default=10000)
parser.add_argument('--overwrite', action='store', type=str, required=False, default='no')
parser.add_argument('--threads', action='store', type=int, required=False, default=12)
args = parser.parse_args()
overwrite = args.overwrite== 'yes'

def write_autocorr(filepath, overwrite=False, intervals=200, max_distance=5000000, points=10000):
    folder, filename = os.path.split(filepath)
    out_path = os.path.join(folder, 'autocorr_intervals_{intervals}_md_{md}_points_{points}.csv'.format(
        intervals=intervals, md=max_distance, points=points))
    if os.path.exists(out_path) and not overwrite:
        return

    with open(filepath, mode='rb') as f:
        process = pickle.load(f)

    records = compute_autocorr_new(process, intervals=intervals, max_distance=max_distance, points=points)

    df = pd.DataFrame(records)
    averages = df.mean(axis=1).reset_index()
    averages.columns = ['interval', 'autocorr']
    averages.to_csv(out_path, index=None)


pool = mp.Pool(processes=args.threads) # how many should we use here?
func = functools.partial(write_autocorr, overwrite=overwrite, intervals=args.intervals, max_distance=args.max_distance,
                         points= args.points)
df = pd.read_csv(args.filepaths)
pool.imap_unordered(func, list(df.filepath))