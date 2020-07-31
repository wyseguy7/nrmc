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

for fi in list(df.filepath):

    print(fi)
    try:
        with open(fi, mode='rb') as f:
            process = pickle.load(f)

        for node, nodedata in process._initial_state.graph.nodes.items():
            process._initial_state.graph.nodes()[node]['Centroid'] = np.array(nodedata['Centroid'], dtype='d')



        fp = os.path.split(fi)[-1]
        out_path = fp.replace('.pkl', '.json')
        com = extract_center_of_mass(process)

        with open(os.path.join(out_folder, fp), mode='w') as f:
            json.dump(com, f)

    except Exception as e:
        print(e)
        continue
