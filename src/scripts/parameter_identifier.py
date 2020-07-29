import sys
import os
import glob
import pandas as pd
import pickle

sys.path.append('/gtmp/etw16/nonreversiblecodebase')

folder = sys.argv[2] # todo check this

files = glob.glob(os.path.join(folder, '*/*.pkl')) + glob.glob(os.path.join(folder, '*/*/*/*.pkl'))
feature_list = []

for fi in files:
    with open(fi, mode='rb') as f:
        process = pickle.load(f)

    features = {
        'class': process.__class__.__name__,
        'beta': process.beta,
        'measure_beta': process.measure_beta,
        'filepath': fi,
        'graph_size': len(process.state.graph.nodes()),
        'score_weights': [i[0] for i in process.score_list] if hasattr(process, 'score_list') else None,
        'score_funcs': [i[1].__name__ for i in process.score_list] if hasattr(process, 'score_list') else None,
        'run_length': len(process.state.move_log),
        'num_districts': len(process.state.color_to_node),
        'minimum_population': process.state.minimum_population,
        'maximum_population': process.state.maximum_population if hasattr(process.state,
                                                                          'maximum_population') else None,
        'center': process.center if hasattr(process, 'center') else None,
        'starting_involution': process._initial_state.involution if hasattr(process._initial_state,
                                                                            'involution') else None,
    }
    feature_list.append(features)

df = pd.DataFrame(feature_list)
df.to_csv("features_out.csv", index=None)
