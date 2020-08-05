import numpy as np
import sys
import pickle
import pandas as pd
import os
import multiprocessing as mp

sys.path.append('/gtmp/etw16/nonreversiblecodebase/')

from src.nrmc.analytics import extract_center_of_mass, center_of_mass_to_polar

df = pd.read_csv(sys.argv[1])
overwrite = False


def func(filepath):
    print(filepath)
    try:
        folder, filename = os.path.split(filepath)
        out_path = os.path.join(folder, 'center_of_mass.csv')
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
        com_polar = center_of_mass_to_polar(com, center)

        df = pd.DataFrame(com_polar)
        df.to_csv(out_path, index=None)

    except Exception as e:
        print(e)
        return

with mp.Pool(processes=4) as pool:
    pool.map(func, list(df.filepath))
