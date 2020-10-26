import argparse
import pickle
import os
import pandas as pd
import multiprocessing as mp
import functools

from nrmc.analytics import calculate_apd

parser = argparse.ArgumentParser()
parser.add_argument('--filepaths', action='store', type=str, required=True, default='features_out.csv')
parser.add_argument('--ideal_pop', action='store', type=float, required=False, default=None)
parser.add_argument('--threads', action='store', type=int, required=False, default=12)
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()




def write_apd(filepath, overwrite=False, ideal_pop=None):

    folder, filename = os.path.split(filepath)
    file_out = 'apd.csv'
    out_path = os.path.join(folder, file_out)
    if os.path.exists(out_path) and not overwrite:
        return

    with open(filepath, mode='rb') as f:
        process = pickle.load(f)

    apd_list = calculate_apd(process, ideal_pop)
    pd.DataFrame(apd_list).to_csv(out_path, index=None)


func = functools.partial(write_apd, overwrite=False, ideal_pop=args.ideal_pop)

df = pd.DataFrame(args.filepaths)

with mp.Pool(processes=args.threads) as pool:
    pool.map(func, list(df.filepath))



