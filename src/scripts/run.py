import sys
import argparse
import os

import pandas as pd

# sys.path.append('/home/grad/etw16/nonreversiblecodebase/') # TODO make this less garbage-y
# sys.path.append('/home/grad/etw16/nonreversiblecodebase/src/legacy/') # TODO make this less garbage-y
# sys.path.append('/home/grad/etw16/nonreversiblecodebase/src/') # TODO make this less garbage-y
sys.path.append('/gtmp/etw16/nonreversiblecodebase')

# sys.path.append('C:\\Users\\wyseg\\nonreversiblecodebase') # TODO make this less garbage-y
# sys.path.append('C:\\Users\\wyseg\\nonreversiblecodebase\\src\\') # TODO make this less garbage-y
# sys.path.append('C:\\Users\\wyseg\\nonreversiblecodebase\\legacy\\') # TODO make this less garbage-y
# sys.path.append('C:\\Users\\wyseg\\nonreversiblecodebase') # TODO make this less garbage-y



from nrmc.state import State

folder_path = '/gtmp/etw16/runs/'

parser = argparse.ArgumentParser()
parser.add_argument('--folder', action='store', type=str, required=False, default=None)
parser.add_argument('--beta', action='store', type=float, required=False, default=1.)
parser.add_argument('--measure_beta', action='store', type=float, required=False, default=2.)
parser.add_argument('--score_func', default=['cut_length'], nargs='+')
parser.add_argument('--score_weights', type=float, default=[1.0], nargs='+')
parser.add_argument('--steps', type=int, default=1000000)
parser.add_argument('--process', type=str, default='single_node_flip')
parser.add_argument('--output_path', type=str, default='/gtmp/etw16/runs/')
parser.add_argument('--ideal_pop', type=float, default=None)
parser.add_argument('--diagonal', action='store_true')
parser.add_argument('--n', type=int, default=40)
parser.add_argument('--involution', type=int, default=1)
parser.add_argument('--num_districts', type=int, default=2)
parser.add_argument('--apd', type=float, default=0.1)
parser.add_argument('--profile', action='store_true')
parser.add_argument('--num_points', default=None, type=int) # for use with star-type lattice
parser.add_argument('--graph_type', default='lattice', type=str)
parser.add_argument('--weight_attribute', default=None, type=str)
parser.add_argument('--involution_max', default=1, type=int)
# parser.add_argument('--tempered', type=str)

args = parser.parse_args()

process_args = {'measure_beta': args.measure_beta,
             'beta': args.beta,
             'score_funcs': args.score_func,
             'score_weights': args.score_weights,
             'folder_path': args.output_path,
             'weight_attribute': args.weight_attribute
                # 'ideal_pop': args.ideal_pop
                }

state_args = {
    'apd': args.apd,
    'involution': args.involution,
    'ideal_pop': args.ideal_pop,
    'num_districts': args.num_districts
}



if args.folder is None:

    if args.graph_type == 'lattice':
        # use a square lattice
        from nrmc.lattice import create_square_lattice
        state_new = create_square_lattice(n=args.n, **state_args)

        if args.diagonal:

            if args.num_districts != 2:
                raise ValueError("--diagonal only implemented for num_districts=2")

            # TODO next time just generate the boundary for this instead
            for node in state_new.graph.nodes():
                centroid = state_new.graph.nodes()[node]['Centroid']
                state_new.node_to_color[node] = int(centroid[0] > centroid[1])

            colors = set(state_new.node_to_color.values())
            state_new.color_to_node = {color: {node for node in state_new.node_to_color if state_new.node_to_color[node] == color} for color in colors}

    elif args.graph_type=='star':
        from nrmc.lattice import create_star
        if args.num_points is None:
            num_points = args.num_districts
        else:
            num_points = args.num_points
        state_new = create_star(n=args.n, num_points=num_points, **state_args)

    else:
        raise ValueError("Must select lattice or star")


else:
    # create state from the relevant folder
    graph_type = args.folder.split(os.path.sep)[-1]
    state_new = State.from_folder(args.folder, graph_type=graph_type, **state_args)

# state_new.ideal_pop = args.ideal_pop
# state_new.involution = args.involution


if args.process == 'single_node_flip':
    from nrmc.single_node_flip import SingleNodeFlip
    process = SingleNodeFlip(state=state_new, **process_args)

elif args.process == 'single_node_flip_tempered':
    from nrmc.single_node_flip import SingleNodeFlipTempered
    process = SingleNodeFlipTempered(state=state_new, **process_args)

elif args.process == 'district_to_district':
    from nrmc.district_to_district import DistrictToDistrictFixed
    process = DistrictToDistrictFixed(state=state_new, **process_args)

elif args.process == 'center_of_mass':
    from nrmc.center_of_mass import CenterOfMassFlow
    process = CenterOfMassFlow(state=state_new, **process_args)

else:
    raise ValueError("Please select a valid process type")

try:

    def f():
        date = str(pd.datetime.today().date())
        for i in range(args.steps):
            process.step()

            if i % 1000000 == 0:
                process.save() #

    if args.profile:
        import cProfile
        cProfile.run('f()')
    else:
        f()

        # no need for logging these anymore
        # if i % 10000 == 0:
        #     f = plt.figure(figsize=(15,8))
        #     draw(process.state.graph, pos={node_id: (process.state.graph.nodes()[node_id]['Centroid'][0],
        #                                              process.state.graph.nodes()[node_id]['Centroid'][1]) for node_id in process.state.graph.nodes()},
        #          node_color=[process.state.node_to_color[i] for i in process.state.graph.nodes()], node_size=100)
        #
        #
        #     filepath = os.path.join(process.folder_path, args.process, '{}_{}_{}.png'.format(date, process.run_id, i))
        #     f.savefig(filepath)
        #     plt.close()

finally:
    # TODO put those heatmaps here so we don't have to do later
    process.save()
