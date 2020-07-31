import sys
sys.path.append('/gtmp/etw16/nonreversiblecodebase/')

import os
import pickle

import pandas as pd
import numpy as np

from src.analytics import compute_autocorr_new
from src.analytics import count_node_colorings

data_points = ['district_1', 'times_contested', 'node_flips']
df = pd.read_csv(sys.argv[1])


def to_array(df, column, n=40):
    mat = np.ndarray(shape=(n, n), dtype='f')

    for row in df.itertuples():
        # mat[int(row.x), int(row.y)] = row.district_1/len(process.state.move_log)
        mat[int(row.x), int(row.y)] = getattr(row, column)

    return mat


for filepath in list(df.filepath):

    folder, filename = os.path.split(filepath)
    out_path = os.path.join(folder, 'field_data.csv')
    if os.path.exists(out_path):
        continue


    with open(filename, mode='rb') as f:
        process = pickle.load(f)



    #     move_log = copy.deepcopy(process.state.move_log[:10000000])
    #     process.state = copy.deepcopy(process._initial_state)
    #     for move in move_log:
    #         update_contested_edges(process.state)
    #         process.state.handle_move(move)

    #     new_state = copy.deepcopy(process._initial_state)
    #     for move in process.state.move_log[:20000000]:
    #         new_state.handle_move(move)

    #     process._initial_state = copy.deepcopy(new_state) # attach initial_state

    #     # new_state.move_log = process.state.move_log[20000000:]
    #     new_state.contested_node_counter = collections.defaultdict(int) # reset contested_node_counter

    #     for move in process.state.move_log[20000000:]:
    #         new_state.handle_move(move)
    #         update_contested_edges(new_state)

    #     process.state = new_state # reattach new state

    data_dict = dict()
    data_dict['records'] = compute_autocorr_new(process, points=10000, max_distance=3000000, intervals=200)

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
        data_dict[col] = to_array(df, col, n=int(np.sqrt(len(process.state.graph.nodes()))))


    df.to_csv(os.path.join(folder, 'field_data.csv'), index=None)

    # data_dict['percentiles'] = np.quantile(list(colors_normalized.values()), np.linspace(0,1,num=100))
    # diff,weights = compute_autocorr_fixed_points(process, points=1000, max_distance=100000)
    # diff, weights = compute_autocorr_bootstrap(process, points=10000, max_distance=5000000)
    # data_dict['diff'] = diff
    # data_dict['weight'] = weights
    # data_dict['df'] = df
    # data_dict['corner_moves'] = [(i, -1)[int(i is None)] for i in find_corner_moves(process)]
