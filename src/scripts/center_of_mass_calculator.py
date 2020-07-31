import numpy as np
import json
import sys
import pickle
import pandas as pd
import os

sys.path.append('/gtmp/etw16/nonreversiblecodebase/')

from src.analytics import extract_center_of_mass

df = pd.read_csv(sys.argv[1])
out_folder = sys.argv[2]
overwrite = False


for filepath in list(df.filepath):

    print(filepath)
    try:
        folder, filename = os.path.split(filepath)
        out_path = os.path.join(folder, 'center_of_mass.csv')
        if os.path.exists(out_path) and not overwrite:
            continue

        with open(filepath, mode='rb') as f:
            process = pickle.load(f)

        for node, nodedata in process._initial_state.graph.nodes.items():
            process._initial_state.graph.nodes()[node]['Centroid'] = np.array(nodedata['Centroid'], dtype='d')

        com = extract_center_of_mass(process)
        df = pd.DataFrame(com)
        df.to_csv(out_path, index=None)

    except Exception as e:
        print(e)
        continue
