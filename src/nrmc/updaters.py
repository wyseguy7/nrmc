import collections

import networkx as nx
import numpy as np


try:
    from .biconnected import calculate_com_inner
    cython_biconnected = True
except ImportError:
    print("No Cython for you!")
    cython_biconnected = False



def contested_edges_naive(state):
    # generate contested edges by testing each edge in the graph. it's brute force and definitely works
    contested = set()

    for edge in state.graph.edges:
        if state.node_to_color[edge[0]] != state.node_to_color[edge[1]]:
            contested.add((min(edge[0], edge[1]), max(edge[0], edge[1])))  # always store small, large
        # if state.graph.nodes[edge[0]]['coloring'] != graph.nodes[edge[1]]['coloring']:
        #     contested.add((min(edge), max(edge))) # always store small, large
    return contested


def calculate_com_naive(state, weight_attribute=None):
    com = {}
    total_weight = {}
    for district_id, nodes in state.color_to_node.items():
        if weight_attribute is None:
            weights = {node_id: 1 for node_id in nodes}
        else:
            weights = {state.graph.nodes()[node_id][weight_attribute] for node_id in nodes}

        total_weight[district_id] = sum(weights.values())
        com[district_id] = np.array([sum([state.graph.nodes()[i]['Centroid'][j]*weights[i] for i in nodes]
                                         )/total_weight[district_id] for j in range(CENTROID_DIM_LENGTH)], dtype='d')
    return com, total_weight


def update_center_of_mass(state):

    if not hasattr(state, 'com_centroid'):
        state.com_centroid, state.com_total_weight = calculate_com_naive(state)
        state.com_updated = state.iteration

    for i in state.move_log[state.com_updated:]:
        if i is not None:
            node_id, old_color, new_color = i
            (state.com_centroid[new_color], state.com_centroid[old_color],
             state.com_total_weight[new_color], state.com_total_weight[old_color]) = calculate_com_one_step(state, i)

    state.com_updated = state.iteration


def create_district_boundary_naive(state):

    district_boundary = collections.defaultdict(set)

    for node_id, color in state.node_to_color.items():

        for neighbor in state.graph.neighbors(node_id):
            if state.node_to_color[neighbor]!= color:
                neighbor_color = state.node_to_color[neighbor]
                district_boundary[(min(color, neighbor_color), max(color, neighbor_color))].add(
                    (min(node_id, neighbor), max(node_id, neighbor)))

    return district_boundary


def parcellation_naive(state):

    parcellation = np.zeros(shape=(len(state.node_to_color), len(state.color_to_node)), dtype=int)
    for node_id, district_id in state.node_to_color.items():
        parcellation[node_id, district_id] = 1
    return parcellation


def update_parcellation(state):

    if not hasattr(state, 'parcellation_matrix'):
        state.parcellation_matrix = parcellation_naive(state)
        state.parcellation_updated = state.iteration

    for move in state.move_log[state.parcellation_updated:]:

        if move is not None:
            node_id, old_color, new_color = move
            state.parcellation_matrix[node_id, old_color] -= 1
            state.parcellation_matrix[node_id, new_color] += 1

    state.parcellation_updated = state.iteration


def update_contested_edges(state):
    if not hasattr(state, 'contested_edges'):
        state.contested_edges = contested_edges_naive(state)
        state.contested_nodes = contested_nodes_naive(state) # maybe we should track this separately?
        state.contested_edges_updated = state.iteration  # set to current iteration

    # this may be an empty list if it's already been updated
    for move in state.move_log[state.contested_edges_updated:]:

        if move is not None:
            node_id, old_color, new_color = move
            # move is provided as (node_id, color_id)
            neighbors = state.graph.edges(node_id)
            # edges to add
            state.contested_edges.update(
                {(min(u, v), max(u, v)) for u, v in neighbors if state.node_to_color[v] != new_color})
            # edges to remove
            state.contested_edges.difference_update(
                {(min(u, v), max(u, v)) for u, v in neighbors if state.node_to_color[v] == new_color})

            neighboring_nodes = set(nx.neighbors(state.graph, node_id))

            # add neighbors that aren't new_color
            state.contested_nodes.update(neighboring_nodes-state.color_to_node[new_color])

            # remove neighbors that are new_color
            state.contested_nodes.difference_update(neighboring_nodes.intersection(state.color_to_node[new_color]))

        if state.log_contested_edges:
            log_contested_edges(state) # separated into function so we can track how expensive this is

    #     # at some point it will be more efficient to just naively reconstruct the contested edges, we should look out for this
    state.contested_edges_updated = state.iteration

def eigen_state_naive(state):

    parcel_matrix = np.zeros(shape=(len(state.node_to_color),len(state.color_to_node)))
    for node_id, district_id in state.node_to_color.items():
        parcel_matrix[node_id, district_id] = 1 # MUST have integer node_id now

    adj_mat_lookup = {}
    for key, matrices in state.matrix_lookup.items():

        adj_mat_list = []
        for matrix in matrices:
            # compute the matrix thing
            adj_mat_list.append(np.matmul(np.matmul(parcel_matrix.T, matrix), parcel_matrix))

        adj_mat_lookup[key] = adj_mat_list

    return adj_mat_lookup


def update_eigen_state(state):


    if not hasattr(state, 'adj_mat'):

        state.adj_mat = eigen_state_naive(state)
        state.adj_mat_updated = state.iteration

    for move in state.move_log[state.adj_mat_updated:]:
        if move is not None:

            node_id, old_color, new_color = move

            for key, matrices in state.adj_mat.items():
                for matrix in matrices:
                    matrix[node_id,old_color] -= k








def perimeter_naive(state):
    # TODO refactor
    dd = collections.defaultdict(int)

    for n0, n1 in state.contested_edges:
        shared_length = state.graph.edges()[(n0,n1)]['border_length']
        dd[state.node_to_color[n0]] += shared_length
        dd[state.node_to_color[n1]] += shared_length

    return dd

def area_naive(state):

    return {district_id: sum(state.graph.nodes()[node_id]['area'] for node_id in state.color_to_node[district_id])
            for district_id in state.color_to_node.keys()}


def update_perimeter_and_area(state):

    # this version assumes that this will get run EVERY time a node is flipped
    update_contested_edges(state) # guarantee contested edges updated before proceeding

    if not hasattr(state, 'district_to_perimeter'):
        state.district_to_perimeter = perimeter_naive(state)
        state.district_to_area = area_naive(state)
        state.perimeter_updated = state.iteration  # set to current iteration

    for move in state.move_log[state.perimeter_updated:]:

        if move is None:
            continue

        node_id, old_color, new_color = move

        for neighbor in state.graph.neighbors(node_id):
            border_length = state.graph.edges[(node_id, neighbor)]['border_length']
            if neighbor in state.color_to_node[new_color]:
                # this border is no longer contested, so subtract from both
                state.district_to_perimeter[new_color] -= border_length
                state.district_to_perimeter[old_color] -= border_length

            elif neighbor in state.color_to_node[old_color]:
                # this border is newly contested - so add to both
                state.district_to_perimeter[old_color] += border_length
                state.district_to_perimeter[new_color] += border_length

            else:
                # we need to increase the perimeter of new_color AND decrease of old color. no change to the perimeter of the 3rd district
                state.district_to_perimeter[new_color] += border_length
                state.district_to_perimeter[old_color] -= border_length


        if state.include_external_border:
            state.district_to_perimeter[old_color] -= state.graph.nodes()[node_id]['external_border']
            state.district_to_perimeter[new_color] += state.graph.nodes()[node_id]['external_border']

        state.district_to_area[old_color] -= state.graph.nodes()[node_id]['area']
        state.district_to_area[new_color] += state.graph.nodes()[node_id]['area']
        state.perimeter_computer.update(node_id, old_color, new_color)

    state.perimeter_updated = state.iteration


def update_population(state):

    if not hasattr(state, 'population_counter'):
        state.population_counter = calculate_population_naive(state)

        if state.ideal_pop is None:
            state.ideal_pop = sum(state.population_counter.values())/len(state.population_counter)

        state.population_deviation = population_balance_naive(state, state.ideal_pop)

    else:
        for move in state.move_log[state.population_counter_updated:]:
            if move is not None:
                node_id, old_color, new_color = move
                state.population_counter[old_color] -= state.graph.nodes()[node_id]['population']
                state.population_counter[new_color] += state.graph.nodes()[node_id]['population']
                # state.population_deviation = population_balance_score(state, move)

    state.population_counter_updated = state.iteration


def check_population(state, node_id, old_color, new_color, minimum=400, maximum=1200):
     return ((state.population_counter[old_color] - state.graph.nodes()[node_id]['population']) > minimum
     and state.population_counter[new_color] + state.graph.nodes()[node_id]['population'] < maximum
     )


def population_balance_naive(state, ideal_pop):
    return np.sqrt(sum((sum([state.graph.nodes()[node]['population'] for node in nodes]) - ideal_pop)**2
                       for nodes in state.color_to_node.values()))


def process_boundary_move(state, node_id, old_color, new_color, neighbor):
    neighbor_color = state.node_to_color[neighbor]
    if neighbor_color != new_color:
        key = (min(neighbor_color, new_color), max(neighbor_color, new_color))
        state.district_boundary[key].add((min(neighbor, node_id), max(neighbor, node_id)))
    if neighbor_color != old_color:
        key = (min(neighbor_color, old_color), max(neighbor_color, old_color))
        state.district_boundary[key].remove((min(neighbor, node_id), max(neighbor, node_id)))


def update_district_boundary(state):

    if not hasattr(state, 'district_boundary'):

        state.district_boundary = create_district_boundary_naive(state)
        state.district_boundary_updated = state.iteration

    moves_to_do = [i for i in state.move_log[state.district_boundary_updated:] if i is not None]

    # TODO we have repeated code in two places, rip out into function

    if len(moves_to_do) < 5:
        for node_id, old_color, new_color in moves_to_do:
            for neighbor in state.graph.neighbors(node_id):
                neighbor_color = state.node_to_color[neighbor]
                key = (min(neighbor_color, new_color), max(neighbor_color, new_color))
                # node_key = (min(neighbor, node_id), max(neighbor))

                if neighbor_color != new_color:
                    state.district_boundary[key].add((min(neighbor, node_id), max(neighbor, node_id)))
                else: # color == new_color
                    node_key = (min(neighbor, node_id), max(neighbor, node_id))
                    key = (min(neighbor_color, old_color), max(neighbor_color, old_color))
                    if node_key in state.district_boundary[key]:
                        state.district_boundary[key].remove(node_key)

    else:
        perturbed_nodes = dict()
        for node_id, old_color, new_color in moves_to_do:

            if node_id in perturbed_nodes:
                perturbed_nodes[node_id][1] = new_color
            else:
                perturbed_nodes[node_id] = (old_color, new_color)

        for node_id, (old_color, new_color) in perturbed_nodes.items():

            for neighbor in state.graph.neighbors(node_id):
                neighbor_color = state.node_to_color[neighbor]
                key = (min(neighbor_color, new_color), max(neighbor_color, new_color))

                if neighbor_color != new_color:
                    state.district_boundary[key].add((min(neighbor, node_id), max(neighbor, node_id)))
                else: # color == new_color
                    # key = (min(neighbor_color, new_color), max(neighbor_color, old_color))
                    state.district_boundary[key].remove((min(neighbor, node_id), max(neighbor, node_id)))

    state.district_boundary_updated = state.iteration


def update_boundary_nodes_naive(state):
    counter = collections.Counter([v for k,v in state.node_to_color.items() if state.graph.nodes()[k]['boundary']])
    return counter


def update_boundary_nodes(state):

    if not hasattr(state, 'boundary_node_counter'):
        state.boundary_node_counter = update_boundary_nodes_naive(state)

    else:
        for move in state.move_log[state.boundary_node_updated:]:
            if move is not None and state.graph.nodes()[move[0]]['boundary']:
                # if we flipped a boundary node
                state.boundary_node_counter[move[1]] -= 1
                state.boundary_node_counter[move[2]] += 1

    state.boundary_node_updated = state.iteration


def compactness_naive(state):

    perimeter_dict = perimeter_naive(state)

    area_dict = {district_id: sum(node['population'] for node in state.graph.nodes() if node in state.graph.color_to_node[district_id])
                 for district_id in state.graph.color_to_node}

    return sum(perimeter_dict[district_id]**2/area_dict[district_id] for district_id in state.graph.color_to_node)


def calculate_com_one_step(state, proposal, weight_attribute=None):
    node_id, old_color, new_color = proposal

    #     com_centroid = copy.deepcopy(state.com_centroid) # ugh, should this function just be side-effecting? how bad is this cost?
    #     total_weight = copy.deepcopy(state.com_total_weight)
    node = state.graph.nodes()[node_id]  # how expensive is this lookup, anyways?

    weight = node[weight_attribute] if weight_attribute is not None else 1

    if cython_biconnected:
        output_new = calculate_com_inner(node['Centroid'], weight, state.com_centroid[new_color],
                                         state.com_total_weight[new_color])
        output_old = calculate_com_inner(node['Centroid'], -weight, state.com_centroid[old_color],
                                         state.com_total_weight[old_color])

        return np.array(output_new[0:2], dtype='d'), np.array(output_old[0:2], dtype='d'), output_new[2], output_old[2]

    else:

        centroid_new_color = (node['Centroid'] * weight + state.com_centroid[new_color] * state.com_total_weight[new_color])/(
                state.com_total_weight[new_color] + weight)
        centroid_old_color = (-node['Centroid'] * weight + state.com_centroid[old_color] * state.com_total_weight[old_color])/(
                state.com_total_weight[old_color] - weight)

        total_weight_new_color = state.com_total_weight[new_color] + weight
        total_weight_old_color = state.com_total_weight[old_color] - weight

        return centroid_new_color, centroid_old_color, total_weight_new_color, total_weight_old_color


def calculate_population_naive(state):
    return {district_id: sum(state.graph.nodes()[node_id]['population']
                             for node_id in state.color_to_node[district_id]) for district_id in state.color_to_node}


def contested_nodes_naive(state):
    contested = set()
    for node_id in state.graph.nodes():
        color = state.node_to_color[node_id]
        if not all(color==state.node_to_color[other_id] for other_id in nx.neighbors(state.graph, node_id)):
            contested.add(node_id)
    return contested


def get_matrix_update(state, proposal):

    update_parcellation(state) # ensure that this is updated


    import copy
    node_id, old_color, new_color = proposal # unpack

    parcellation = copy.copy(state.parcellation_matrix) # parcellation is n times p
    # this will get expensive, should avoid? but probably not worse than anything else we're doing
    parcellation[node_id, old_color] -= 1
    parcellation[node_id, new_color] += 1

    return get_matrix_naive(state, parcellation)

def get_matrix_naive(state, parcellation):
    matrix_update = dict()

    for group_id, adj_mat_lookup in state.full_adj_lookup.items():

        adj_mat_update = dict()
        for graph_id, adj_mat in adj_mat_lookup.items():
            adj_mat_update[graph_id] = parcellation.T @ adj_mat @ parcellation
        matrix_update[group_id] = adj_mat_update
    return matrix_update

def fast_get_matrix_update(state, proposal):

    import copy
    node_id, old_color, new_color = proposal # unpack

    parcellation = copy.copy(state.parcellation_matrix) # parcellation is n times p,
    parcellation[node_id, old_color] -= 1
    parcellation[node_id, new_color] += 1

    matrix_update = copy.copy(state.matrix_lookup)
    for group_id, adj_mat_lookup in matrix_update.items():
        group_full_adj_mats = state.full_adj_lookup[group_id] # TODO add this to state

        for graph_id, adj_mat in adj_mat_lookup.items():
            graph_adj = group_full_adj_mats[graph_id]

            s_1 = graph_adj[node_id, parcellation[:, new_color]].sum()
            s_2 = graph_adj[node_id, parcellation[:, old_color]].sum()
            s_3 = graph_adj[node_id, node_id]

            adj_mat[new_color, new_color] += 2*s_1 - s_3
            off_diag_sum = s_2-s_1

            adj_mat[old_color, new_color] += off_diag_sum
            adj_mat[new_color, old_color] += off_diag_sum
            adj_mat[old_color, old_color] += s_3 - 2*s_2

            # TODO definitely check this section for bugs

    return matrix_update


def log_contested_edges(state):
    # maybe manage an array so we don't need to use these horrible loops
    for edge in state.contested_edges:
        # contested_nodes = set(itertools.chain(*state.contested_edges)) # this is horrible, maybe start tracking contested nodes?
        state.contested_edge_counter[edge] += 1

    for node in state.contested_nodes:
        state.contested_node_counter[node] += 1


CENTROID_DIM_LENGTH = 2 # TODO do we need a settings.py file?