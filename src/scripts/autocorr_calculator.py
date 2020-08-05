import sys
import os
import pickle
import multiprocessing as mp
import functools
import argparse

import pandas as pd

sys.path.append('/gtmp/etw16/nonreversiblecodebase/')
from src.nrmc.analytics import compute_autocorr_new

data_points = ['district_1', 'times_contested', 'node_flips']

parser = argparse.ArgumentParser()
parser.add_argument('--filepaths', action='store', type=str, required=True, default='features_out.csv')
parser.add_argument('--intervals', action='store', type=int, required=False, default=200)
parser.add_argument('--max_distance', action='store', type=int, required=False, default=5000000)
parser.add_argument('--points', action='store', type=int, required=False, default=10000)
parser.add_argument('--overwrite', action='store', type=str, required=False, default='no')
parser.add_argument('--threads', action='store', type=int, required=False, default=12)
parser.add_argument('--truncate', action='store', type=int, required=False, default=0)
args = parser.parse_args()
overwrite = args.overwrite== 'yes'

def write_autocorr(filepath, overwrite=False, intervals=200, max_distance=5000000, points=10000, truncate=0):
    print(filepath)
    folder, filename = os.path.split(filepath)
    truncate_str = '' if truncate == 0 else '_truncate_{}'.format(truncate)

    out_path = os.path.join(folder, 'autocorr_intervals_{intervals}_md_{md}_points_{points}{truncate}.csv'.format(
        intervals=intervals, md=max_distance, points=points, truncate=truncate_str))
    if os.path.exists(out_path) and not overwrite:
        print('skipping')
        return

    with open(filepath, mode='rb') as f:
        process = pickle.load(f)
    print('loaded')

    if truncate != 0:
        process.state.move_log = process.state.move_log[:truncate]

    records = compute_autocorr_new(process, intervals=intervals, max_distance=max_distance, points=points)
    print('sampled')
    df = pd.DataFrame(records)
    averages = df.mean(axis=1).reset_index()
    averages.columns = ['interval', 'autocorr']
    averages.to_csv(out_path, index=None)


func = functools.partial(write_autocorr, overwrite=overwrite, intervals=args.intervals, max_distance=args.max_distance,
                         points = args.points, truncate=args.truncate)
df = pd.read_csv(args.filepaths)
print(len(df))
# [func(i) for i in list(df.filepath)]
def safety(filepath):
    try:
        return func(filepath)
    except Exception as e:
        print(e)


with mp.Pool(processes=args.threads) as pool:
    pool.map(safety, list(df.filepath))
