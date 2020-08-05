import sys
sys.path.append('/gtmp/etw16/nonreversiblecodebase/')

import os
import pickle
import argparse
import multiprocessing as mp
import copy

import pandas as pd
import numpy as np

from src.nrmc.analytics import count_node_colorings

data_points = ['district_1', 'times_contested', 'node_flips']

parser = argparse.ArgumentParser()

parser.add_argument('--filepaths', action='store', type=str, required=True, default='features_out.csv')
parser.add_argument('--threads', action='store', type=int, required=False, default=12)
parser.add_argument('--truncate', action='store', type=int, required=False, default=0)
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()

df = pd.read_csv(args.filepaths)


def to_array(df, column, n=40):
    mat = np.ndarray(shape=(n, n), dtype='f')

    for row in df.itertuples():
        # mat[int(row.x), int(row.y)] = row.district_1/len(process.state.move_log)
        mat[int(row.x), int(row.y)] = getattr(row, column)

    return mat

def make_heatmap(filepath):
    print(filepath)

    folder, filename = os.path.split(filepath)
    truncate_str = '' if args.truncate == 0 else '_truncate_{}'.format(args.truncate)
    out_path = os.path.join(folder, 'field_data{}.csv'.format(truncate_str))
    if os.path.exists(out_path) and not args.overwrite:
        return


    with open(filepath, mode='rb') as f:
        process = pickle.load(f)

    if args.truncate != 0:
        from src import update_contested_edges
        process.state.move_log = process.state.move_log[:args.truncate]
        initial_state = copy.deepcopy(process._initial_state)
        for move in process.state.move_log:

            update_contested_edges(process._initial_state)
            process._initial_state.handle_move(move)

        process.state = copy.deepcopy(process._initial_state)
        process._initial_state = initial_state


    if len(process.state.move_log) == 0:
        return

    n = int(np.sqrt(len(process.state.graph.nodes())))
    if n**2 != len(process.state.graph.nodes()):
        # graph size isn't a perfect square
        return # this is not sensible to do, skip this one

    data_dict = dict()

    node_colorings = count_node_colorings(process)

    colors_1 = {k: v[1] for k, v in node_colorings.items()}
    colors_normalized = {k: v / (len(process.state.move_log))
                         for k, v in colors_1.items()}

    def count_node_flips(process):
        import collections
        # count the number of times each node was flipped
        return collections.Counter([i[0] for i in process.state.move_log if i is not None])


    node_flips = count_node_flips(process)
    node_flips_normalized = {k: v / sum(node_flips.values())
                             for k, v in node_flips.items()}
    node_flips_normalized.update({k: 0. for k in process.state.graph.nodes() if k not in node_flips_normalized})

    idx_list = list(node_flips_normalized.keys())  # assure ordering
    x = [process.state.graph.nodes()[i]['Centroid'][0] for i in idx_list]
    y = [process.state.graph.nodes()[i]['Centroid'][1] for i in idx_list]

    df = pd.DataFrame(
        {"node_flips": pd.Series([node_flips_normalized[i] for i in idx_list], index=idx_list, dtype='float64'),
         "district_1": pd.Series([colors_normalized[i] for i in idx_list], index=idx_list, dtype='float64'),
         "times_contested": pd.Series([process.state.contested_node_counter[i] for i in idx_list],
                                      index=idx_list, dtype='float64'),
         'x': pd.Series(x, index=idx_list),
         'y': pd.Series(y, index=idx_list)

         })
    for col in data_points:
        data_dict[col] = to_array(df, col, n=n)


    df.to_csv(out_path, index=None)

with mp.Pool(processes=args.threads) as pool:
    pool.map(make_heatmap, list(df.filepath))
