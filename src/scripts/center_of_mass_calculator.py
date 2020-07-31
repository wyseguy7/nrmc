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

    with open(fi, mode='rb') as f:
        try:
            process = pickle.load(f)
        except:
            continue

    fp = os.path.split(fi)[-1]
    out_path = fp.replace('.pkl', '.json')
    com = extract_center_of_mass(process)

    with open(os.path.join(out_folder, fp), mode='w') as f:
        json.dump(com)

