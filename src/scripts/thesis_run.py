import argparse
import os
import pandas as pd
import numpy as np
import collections
from nrmc.lattice import create_square_lattice
from nrmc.car_gibbs import CenterOfMassFlowGibbs, GibbsMixing
from nrmc.state import State


def graph_laplacian_naive(self):

    lap = np.zeros(shape=(len(self.node_to_color), len(self.node_to_color)), dtype='int')
    for node_id, color in self.node_to_color.items():
        neighbors = list(self.graph.neighbors(node_id))

        lap[node_id, node_id] = len(neighbors)

        for neighbor in neighbors:
            if self.node_to_color[neighbor] == color:
                lap[node_id, neighbor] = -1
                
    return lap


def coloring_balocchi(lat, sep=3):
    
    node_to_color = {node : 0 for node in lat.node_to_color}
    
    myset = {70, 69, 89, 90, 110, 109, 130, 129, 91, 111, 88, 108}    
    for node in node_to_color:
        cent = lat.graph.nodes()[node]['Centroid']
        if cent[0] >= 10:
            if cent[1] >= 10:
                node_to_color[node] = 1
            else:
                node_to_color[node] = 2
        else:
            if node in myset:
                node_to_color[node] = 3
    
    color_to_grand_mean = {0: -sep, 1: 0, 2: sep, 3: 0}
    return node_to_color, color_to_grand_mean



def get_simulation_data(rho=0.9, sigma=0.1, sep=3):
    
    lat = create_square_lattice(n=20, num_districts=3)
        
    obs_per_node = 12
    
    # color_to_grand_mean = {0: -10, 1: 10}

    # node_to_color, coloring_to_grand_mean = coloring_tripartite(lat)
    node_to_color, color_to_grand_mean = coloring_balocchi(lat, sep=sep)

    d = collections.defaultdict(set)
    for node_id, district_id in lat.node_to_color.items():
        d[district_id].add(node_id)

    lat.color_to_node = d
    lat.node_to_color = node_to_color
    
    grand_mean_vec = np.array([[color_to_grand_mean[lat.node_to_color[i]] for i in range(len(lat.node_to_color))]])
    normal_noise = np.random.normal(size=len(lat.node_to_color))
    
    W = graph_laplacian_naive(lat)
    
    Sigma  = rho*W + (1-rho)*np.identity(len(lat.node_to_color))
    chol = np.linalg.cholesky(np.linalg.inv(Sigma))
    
    alpha = grand_mean_vec + chol @ normal_noise
    obs_noise = np.random.normal(size=(alpha.shape[1], obs_per_node)) * np.sqrt(sigma)
    Y = alpha.T @ np.ones(shape=(1, obs_per_node)) + obs_noise
    return alpha.T, Y, lat
    
    
    
    # the true cluster-based mean, for data generation purposes
    # X = get_data_matrix(node_to_color, color_to_beta, data_true)

folder_path = '/gtmp/etw16/thesis_runs/'

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
parser.add_argument('--num_districts', type=int, default=4)
parser.add_argument('--apd', type=float, default=0.1)
parser.add_argument('--profile', action='store_true')
parser.add_argument('--num_points', default=None, type=int) # for use with star-type lattice
parser.add_argument('--graph_type', default='lattice', type=str)
parser.add_argument('--weight_attribute', default=None, type=str)
parser.add_argument('--involution_max', default=1, type=int)
parser.add_argument('--lambda_scalar', default=100., type=float)
parser.add_argument('--rho', default=0.9, type=float)
parser.add_argument('--sep', default=3.0, type=float)
parser.add_argument('--sigma', default=0.1, type=float)

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
    'num_districts': args.num_districts,
    'rho': args.rho,
    'lambda_scalar': args.lambda_scalar
}



alpha, Y, lat = get_simulation_data(rho=args.rho, sep=args.sep, sigma=args.sigma)
X = np.ones(shape=(12, 1))
lat_learner = State.from_object(Y, X, lat.graph, **state_args)
lat_learner.minimum_population = 5. # we should fix this
lat_learner.maximum_population = 300.


# else:
#     # create state from the relevant folder
#     graph_type = args.folder.split(os.path.sep)[-1]
#     state_new = State.from_folder(args.folder, graph_type=graph_type, **state_args)
#
# state_new.ideal_pop = args.ideal_pop
# state_new.involution = args.involution

# process = SingleNodeFlipGibbs(lat_learner, score_funcs=['cut_length', 'car_model'],
#                                score_weights=[1.0, 0.005], measure_beta=1.8)

# for i in tqdm(range(10000)):
#     process.step()


if args.process == 'single_node_flip':
    process = GibbsMixing(state=lat_learner, **process_args)

elif args.process == 'center_of_mass':
    process = CenterOfMassFlowGibbs(state=lat_learner, **process_args)

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


finally:
    # TODO put those heatmaps here so we don't have to do later
    process.save()




