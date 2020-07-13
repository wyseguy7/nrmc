import sys
import argparse
from matplotlib import pyplot as plt
import pandas as pd
import os
from networkx import draw

sys.path.append('/home/grad/etw16/nonreversiblecodebase/') # TODO make this less garbage-y
sys.path.append('/home/grad/etw16/nonreversiblecodebase/src/legacy/') # TODO make this less garbage-y
sys.path.append('/home/grad/etw16/nonreversiblecodebase/src/') # TODO make this less garbage-y

sys.path.append('C:\\Users\\wyseg\\nonreversiblecodebase') # TODO make this less garbage-y
sys.path.append('C:\\Users\\wyseg\\nonreversiblecodebase\\src\\') # TODO make this less garbage-y
sys.path.append('C:\\Users\\wyseg\\nonreversiblecodebase\\legacy\\') # TODO make this less garbage-y
sys.path.append('C:\\Users\\wyseg\\nonreversiblecodebase') # TODO make this less garbage-y

from src.state import State

folder_path = '/gtmp/etw16/runs/'

parser = argparse.ArgumentParser()
parser.add_argument('--folder', action='store', type=str, required=False, default=None)
parser.add_argument('--beta', action='store', type=float, required=False, default=2.)
parser.add_argument('--measure_beta', action='store', type=float, required=False, default=1.)
parser.add_argument('--score_func', default=['cut_length'], nargs='+')
parser.add_argument('--score_weights', type=float, default=[1.0], nargs='+')
parser.add_argument('--steps', type=int, default=1000000)
parser.add_argument('--process', type=str, default='single_node_flip')
parser.add_argument('--output_path', type=str, default='/gtmp/etw16/runs/')
parser.add_argument('--ideal_pop', type=float, default=None)
# parser.add_argument('--tempered', type=str)

args = parser.parse_args()


if args.folder is None:
    # use a square lattice
    from src.scripts.lattice import state_new

else:
    # create state from the relevant folder
    state_new = State.from_folder(args.folder)

state_new.ideal_pop = args.ideal_pop


args_dict = {'measure_beta': args.measure_beta,
             'beta': args.beta,
             'score_funcs': args.score_func,
             'score_weights': args.score_weights,
             'folder_path': args.output_path,
             # 'ideal_pop': args.ideal_pop
             }

if args.process == 'single_node_flip':
    from src.single_node_flip import SingleNodeFlip
    process = SingleNodeFlip(state=state_new, **args_dict)

elif args.process == 'single_node_flip_tempered':
    from src.single_node_flip import SingleNodeFlipTempered
    process = SingleNodeFlipTempered(state=state_new, **args_dict)

elif args.process == 'district_to_district':
    from src.district_to_district import DistrictToDistrictTempered
    process = DistrictToDistrictTempered(state=state_new, **args_dict)

elif args.process == 'center_of_mass':
    from src.center_of_mass import CenterOfMassFlow
    process = CenterOfMassFlow(state=state_new, **args_dict)

else:
    raise ValueError("Please select a valid process type")

try:
    date = str(pd.datetime.today().date())
    for i in range(args.steps):
        process.step()

        if i % 10000 == 0:
            f = plt.figure(figsize=(15,8))
            draw(process.state.graph, pos={node_id: (process.state.graph.nodes()[node_id]['Centroid'][0],
                                                     process.state.graph.nodes()[node_id]['Centroid'][1]) for node_id in process.state.graph.nodes()},
                 node_color=[process.state.node_to_color[i] for i in process.state.graph.nodes()], node_size=100)


            filepath = os.path.join(process.folder_path, args.process, '{}_{}_{}.png'.format(date, process.run_id, i))
            f.savefig(filepath)
            plt.close()

finally:
    # TODO put those heatmaps here so we don't have to do later
    process.save()
