import os
import multiprocessing as mp
import functools
import argparse
import numpy as np
from collections import Counter, defaultdict

import pandas as pd
import time

parser = argparse.ArgumentParser()
# NOTE different behavior here - filepaths is a LIST of filepaths
parser.add_argument('--filepaths', action='store', nargs='+', type=str, required=True, default='features_out.csv')
parser.add_argument('--overwrite', action='store', type=str, required=False, default='no')
parser.add_argument('--threads', action='store', type=int, required=False, default=12)

args = parser.parse_args()
overwrite = args.overwrite == 'yes'

def quantize(x):
    return round(2*x, 2)/2 # round to half-percentage point


def collect_var(filepath_csv, overwrite=False, thinning_iterval=10000):

    checkpoint = time.time()

    df_filepaths = pd.read_csv(filepath_csv)
    fo, fi = os.path.split(filepath_csv)
    out_path = os.path.join(fo, 'phi_marginals'+fi)

    if os.path.exists(out_path) and not overwrite:
        return
    my_list = []
    for filepath in df_filepaths['filepath']:
        fi_wins = os.path.join(os.path.split(filepath)[0], 'marginals.csv')
        df_fi = pd.read_csv(fi_wins)
        my_list.append(df_fi)
    # print('pang')
    print("Finished loading data in {:.2f} seconds ".format(time.time()-checkpoint))
    checkpoint = time.time()


    num_districts = len(my_list[0].columns)

    district_list = []
    for district_idx in range(num_districts):

        thingy = [df.iloc[:,district_idx].apply(quantize) for df in my_list]
        df = pd.concat(thingy, axis=1)
        district_list.append(df)

    # now, we have quantized picture of each

    pibar_total = [Counter(np.ravel(df.values)) for df in district_list] # make this work
    pibar_list = [{k:v/sum(d.values()) for k,v in d.items()} for d in pibar_total] # normalize by counts

    print(pibar_list)

    print("Finished calculating pibar in {:.2f} seconds ".format(time.time()-checkpoint))
    # checkpoint = time.time()
    K = len(my_list) # number of chains
    N = len(my_list[0])
    tv_dist = np.zeros(shape=(N, K, num_districts))
    for district_idx in range(num_districts):

        pibar = pibar_list[district_idx]
        df = district_list[district_idx]

        hist_list = [defaultdict(int) for i in range(K)]
        # counter = 0
        for i in range(len(df)):
            # counter += 1
            for k in range(K):

                idx = df.iloc[i,k]
                hist_list[k][idx] += 1
                print(idx)
                tv_dist[i,k,district_idx] = 0.5*np.sum([abs(hist_list[k][key]-v) for key, v in pibar.items()])/(i+1)

            if i % 1000000 == 0:
                print("Finished calculating 1000000 iterations in {:.2f} seconds ".format(time.time() - checkpoint))
                checkpoint = time.time()
                print(hist_list[0])

    tv_dist_col = tv_dist.mean(axis=2) # take the mean over tv distance for each
    df_out = pd.DataFrame(tv_dist_col)
    df_out.to_csv(out_path, index=None)


func = functools.partial(collect_var, overwrite=overwrite)

def safety(filepath):
    try:
        return func(filepath)
    except Exception as e:
        print(e)


with mp.Pool(processes=args.threads) as pool:
    pool.map(func, list(args.filepaths))
