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
# parser.add_argument('--internal_threads', action='store', type=int, required=False, default=1)
parser.add_argument('--thinning_interval', action='store', type=int, required=False, default=10000)
parser.add_argument('--burn_in', action='store', type=int, required=False, default=0)

args = parser.parse_args()
overwrite = args.overwrite == 'yes'

def quantize(x):
    return round(2*x, 2)/2 # round to half-percentage point

def compute_tv_individual(k, district_idx, pibar, column, thinning_interval=10000):
    output = np.ndarray(shape=(int(len(column)/thinning_interval)))
    # pibar = pibar_list[district_idx]
    # column = district_list[district_idx].iloc[:,k] # obtain the k-th column from the correct district idx

    full_count = Counter()
    for i in range(len(output)):

        count = Counter(column.iloc[ (i*thinning_interval): ((i+1)*thinning_interval)]) # this is fine
        full_count.update(count)

        output[i] = 0.5 * np.sum([abs(full_count[key] / ((i+1)*thinning_interval) - v) for key, v in pibar.items()])

        if i % 100 == 0:
            print(i*thinning_interval)



    return full_count, output

def make_initial_counter(df):
    return Counter(np.ravel(df.values))

def read_initial_csv(filepath, burn_in=0):
    fi_wins = os.path.join(os.path.split(filepath)[0], 'marginals.csv')
    df_fi = pd.read_csv(fi_wins).iloc[burn_in:]
    return df_fi.apply(quantize)


read_func = functools.partial(read_initial_csv, burn_in=args.burn_in)
func = functools.partial(compute_tv_individual, thinning_interval=args.thinning_interval)

def collect_var(filepath_csv, overwrite=False, thinning_interval=10000, threads=1):

    checkpoint = time.time()

    df_filepaths = pd.read_csv(filepath_csv)
    fo, fi = os.path.split(filepath_csv)
    out_path = os.path.join(fo, 'phi_marginals'+fi)

    if os.path.exists(out_path) and not overwrite:
        return
    # my_list = []
    with mp.Pool(processes=threads) as pool:
        my_list = pool.map(read_func, df_filepaths['filepath'])

    # for filepath in df_filepaths['filepath']:
    #     fi_wins = os.path.join(os.path.split(filepath)[0], 'marginals.csv')
    #     df_fi = pd.read_csv(fi_wins)
    #     # df_fi = df_fi.iloc[::10000,:] # TODO rip out once finished debugging
    #     my_list.append(df_fi)
    # print('pang')
    print("Finished loading data in {:.2f} seconds ".format(time.time()-checkpoint))
    checkpoint = time.time()

    num_districts = len(my_list[0].columns)

    district_list = []
    for district_idx in range(num_districts):

        # thingy = [df.iloc[:,district_idx].apply(quantize) for df in my_list]
        thingy = [df.iloc[:, district_idx] for df in my_list]
        df = pd.concat(thingy, axis=1)
        district_list.append(df)

    with mp.Pool(processes=threads) as pool:
        pibar_total = pool.map(make_initial_counter, district_list)
    # pibar_total = [Counter(np.ravel(df.values)) for df in district_list] # make this work
    pibar_list = [{k:v/sum(d.values()) for k,v in d.items()} for d in pibar_total] # normalize by counts
    print(pibar_list)

    pibar_dict = dict()
    for i in range(len(pibar_list)):
        pibar_dict[i] = pibar_list[i] # column district_idx,

    pibar = pd.DataFrame(pibar_dict)
    pibar.to_csv(os.path.join(fo, "pibar"+fi))


    print("Finished calculating pibar in {:.2f} seconds ".format(time.time()-checkpoint))
    # checkpoint = time.time()
    K = len(my_list) # number of chains
    N = len(my_list[0])
    # tv_dist = np.zeros(shape=(N, K, num_districts))



    tv_dist = np.zeros(shape=(int(len(my_list[0])/thinning_interval), K, num_districts))
    print(tv_dist.shape)
    del my_list # release memory on these, we don't need them

    to_process = []
    for k in range(K):
        for district_idx in range(num_districts):
            to_process.append((k,district_idx, pibar_list[district_idx], district_list[district_idx].iloc[:, k]))

    with mp.Pool(processes=threads) as pool:
        results  = pool.starmap(func, to_process)

    full_count_dict = defaultdict(dict)

    for (k, district_idx, pibar, column), (full_count, output) in zip(to_process, results):
        tv_dist[:, k, district_idx] = output

        full_count_dict[k][district_idx] = full_count

    for k, count_dict in full_count_dict.items():

        hist_df = pd.DataFrame(count_dict)
        hist_df.to_csv(os.path.join(fo, 'phi_marginals_chain{}'.format(k)+fi))

    tv_dist_col = tv_dist.mean(axis=2) # take the mean over tv distance for each district_idx
    df_out = pd.DataFrame(tv_dist_col)
    df_out.to_csv(out_path, index=None)

    # output largest deviance histogram for each district
    max_dev_chains_dict = dict()
    for district_idx in range(num_districts):

        chain_idx = np.argmax(tv_dist[-1, :, district_idx])
        max_dev_chains_dict[district_idx] = full_count_dict[chain_idx][district_idx]

    max_dev_chains = pd.DataFrame(max_dev_chains_dict)
    max_dev_chains.to_csv(os.path.join(fo, "max_dev_"+fi))







        # # for i in range(len(df)):
        #     # counter += 1
        #         # print(idx)
        #     idx = df.iloc[i, k]
        #         # hist_list[k][idx] += 1
        #     hist += 1
        #
        #     if i % thinning_interval == 0:
        #
        #         # tv_dist[i,k,district_idx] = 0.5*np.sum([abs(hist_list[k][key-*]/(i+1)-v) for key, v in pibar.items()])
        #         tv_dist[i, k, district_idx] = 0.5 * np.sum(
        #             [abs(hist_list[k][key] / (i+1) - v) for key, v in pibar.items()])
    #
    # for district_idx in range(num_districts):
    #
    #     pibar = pibar_list[district_idx]
    #     df = district_list[district_idx]
    #
    #     hist_list = [defaultdict(int) for i in range(K)]
    #     # counter = 0
    #     for i in range(len(df)):
    #         # counter += 1
    #             # print(idx)
    #         for k in range(K):
    #             idx = df.iloc[i, k]
    #             hist_list[k][idx] += 1
    #
    #         if i % thinning_interval == 0:
    #
    #             # tv_dist[i,k,district_idx] = 0.5*np.sum([abs(hist_list[k][key-*]/(i+1)-v) for key, v in pibar.items()])
    #             tv_dist[i, k, district_idx] = 0.5 * np.sum(
    #                 [abs(hist_list[k][key] / (i+1) - v) for key, v in pibar.items()])
    #
    #         if i % 1000000 == 0:
    #             print("Finished calculating 1000000 iterations in {:.2f} seconds ".format(time.time() - checkpoint))
    #             checkpoint = time.time()
    #             print(hist_list[0])
    #             print(hist_list[1])

for filepath in args.filepaths:
    collect_var(filepath, overwrite=overwrite, thinning_interval=args.thinning_interval, threads=args.threads)


# func = functools.partial(collect_var, overwrite=overwrite, thinning_interval=args.thinning_interval, threads=args.internal_threads)

# def safety(filepath):
#     try:
#         return func(filepath)
#     except Exception as e:
#         print(e)
#
#
# with mp.Pool(processes=args.threads) as pool:
#     pool.map(func, list(args.filepaths))
