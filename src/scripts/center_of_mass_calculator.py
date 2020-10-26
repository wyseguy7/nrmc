import numpy as np
import sys
import pickle
import pandas as pd
import os
import multiprocessing as mp
import argparse

sys.path.append('/gtmp/etw16/nonreversiblecodebase/')

from src.nrmc.analytics import extract_center_of_mass, center_of_mass_to_polar

parser = argparse.ArgumentParser()
parser.add_argument('--filepaths', action='store', type=str, required=True, default='features_out.csv')
parser.add_argument('--polar', action='store_true')
parser.add_argument('--threads', action='store')
args = parser.parse_args()
overwrite = args.overwrite== 'yes'

files = pd.read_csv(args.filepaths)
# overwrite = False


def func(filepath, polar=True):
    print(filepath)
    try:
        folder, filename = os.path.split(filepath)

        file_out = 'center_of_mass_{}.csv'.format(('', 'polar')[polar])
        out_path = os.path.join(folder, file_out)
        if os.path.exists(out_path) and not overwrite:
            return

        with open(filepath, mode='rb') as f:
            process = pickle.load(f)

        for node, nodedata in process._initial_state.graph.nodes.items():
            process._initial_state.graph.nodes()[node]['Centroid'] = np.array(nodedata['Centroid'], dtype='d')

        com = extract_center_of_mass(process)

        weight_attribute = None
        center = np.array([0, 0], dtype='d')
        for i in range(2):  # will this always be R2?
            center[i] = sum(nodedata['Centroid'][i] *
                            (nodedata[weight_attribute] if weight_attribute is not None else 1)
                            for node, nodedata in process.state.graph.nodes.items()) / len(process.state.graph.nodes())

        if polar:
            com_polar = center_of_mass_to_polar(com, center)
            df = pd.DataFrame(com_polar)

        else:
            df = pd.DataFrame(com) # dict with centres of mass

        df.to_csv(out_path, index=None)


    except Exception as e:
        print(e)
        return

with mp.Pool(processes=args.threads) as pool:
    pool.map(func, list(files.filepath))
