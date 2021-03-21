import numpy as np
import collections
from nrmc.lattice import create_square_lattice
from nrmc.single_node_flip import SingleNodeFlipGibbs
from nrmc.state import State
from tqdm import tqdm


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


def coloring_tripartite(lat):
    node_to_color = {node : 1 for node in lat.node_to_color}
#     for node in node_to_color:
#         cent = lat.graph.nodes()[node]['Centroid']
#         if cent[0] >= 10:
#             node_to_color[node] = 0
    color_to_grand_mean = {0: -10, 1: 0, 2: 10}

    for node in node_to_color:
        cent = lat.graph.nodes()[node]['Centroid']
        if cent[0] >= 3 and cent[0] < 8 and cent[1] >= 3 and cent[1] < 8:
            node_to_color[node] = 0
        elif cent[0] >= 12 and cent[0] < 17 and cent[1] >= 12 and cent[1] < 17:
            node_to_color[node] = 2
    
    return node_to_color



def get_simulation_data(rho=0.9, sigma=0.1):
    
    lat = create_square_lattice(n=20, num_districts=3)
        
    obs_per_node = 12
    
    # color_to_grand_mean = {0: -10, 1: 10}

    # node_to_color, coloring_to_grand_mean = coloring_tripartite(lat)
    node_to_color, color_to_grand_mean = coloring_balocchi(lat, sep=4)

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
    

alpha, Y, lat = get_simulation_data()
X = np.ones(shape=(12,1))
lat_learner = State.from_object(Y, X, lat.graph, # node_to_color=lat.node_to_color,  
                                rho=0.9, num_districts=4, apd=1., lambda_scalar=10.)
lat_learner.minimum_population = 5.
lat_learner.maximum_population = 300.

process = SingleNodeFlipGibbs(lat_learner, score_funcs=['cut_length', 'car_model'], 
                               score_weights=[1.0, 0.005], measure_beta=1.8)

for i in tqdm(range(10000)):
    process.step()


