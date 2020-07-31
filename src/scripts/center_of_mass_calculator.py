import json
import sys
import pickle
import pandas as pd
import os

from src.analytics import extract_center_of_mass


sys.path.append('/gtmp/etw16/nonreversiblecodebase/')


df = pd.read_csv(sys.argv[1])

for fi in list(df.filepath):

    with open(fi, mode='rb') as f:
        try:
            process = pickle.load(f)
        except:
            continue

    fp = os.path.split(fi)[-1]
    out_path = fp.replace('.pkl', '.json')
    com = extract_center_of_mass(process)

    with open(fp, mode='w') as f:
        json.dump(com)

