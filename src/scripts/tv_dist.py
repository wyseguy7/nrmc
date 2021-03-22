import os
import multiprocessing as mp
import functools
import argparse
import numpy as np

from networkx.drawing.nx_pylab import draw
from matplotlib import pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
# NOTE different behavior here - filepaths is a LIST of filepaths
parser.add_argument('--filepaths', action='store', nargs='+', type=str, required=True, default='features_out.csv')
parser.add_argument('--overwrite', action='store', type=str, required=False, default='no')
parser.add_argument('--threads', action='store', type=int, required=False, default=12)

args = parser.parse_args()
overwrite = args.overwrite == 'yes'

def collect_var(filepath_csv, states=6, overwrite=False):

    df_filepaths = pd.read_csv(filepath_csv)
    fo, fi = os.path.split(filepath_csv)
    out_path = os.path.join(fo, 'phi_'+fi)

    if os.path.exists(out_path) and not overwrite:
        return
    my_list = []
    for filepath in df_filepaths['filepath']:
        fi_wins = os.path.join(os.path.split(filepath)[0], 'wins.csv')
        df_fi = pd.read_csv(fi_wins)
        my_list.append(df_fi)
    # print('pang')
    df_all = pd.concat(my_list, axis=1).dropna() # TODO check this
    # TODO drop extra rows - truncate to a maximum value


    n = len(df_all) # number of iterations
    m = len(df_all.columns) # number of chains

    phi = np.ndarray(shape=(n,m))

    pibar_total = np.zeros(shape=(states,))
    chain_i = [np.zeros(shape=(states,)) for _ in range(m)]

    for i in range(n):
        for j in range(m):
            x = int(df_all.iloc[i, j])  # guarantee int
            pibar_total[x] += 1

    pibar_total /= (n*m) # divide by n

    for i in range(n):
        for j in range(m):
            x = int(df_all.iloc[i,j]) # guarantee int

            chain_i[j][x] += 1

        # compute phi
        for j in range(m):
            phi[i,j] = max(abs(pibar_total-chain_i[j]/(i+1))) # normalization happens here

        if i % 1000000 == 0:
            print(i)



    phi_mean = np.mean(phi, axis=1)
    phi_var = np.var(phi, axis=1)
    df_out = pd.DataFrame({"phi_mean": phi_mean, "phi_var": phi_var})

    df_out.to_csv(out_path, index=None)



func = functools.partial(collect_var, overwrite=overwrite)

def safety(filepath):
    try:
        return func(filepath)
    except Exception as e:
        print(e)


with mp.Pool(processes=args.threads) as pool:
    pool.map(func, list(args.filepaths))
