import numpy as np

from .updaters import get_matrix_update, eigen_naive

try:
    from .biconnected import biconnected_dfs, dot_product, calculate_com_inner, PerimeterComputer
    cython_biconnected = True
except ImportError:
    cython_biconnected = False


def compactness_score(state, proposal):

    if cython_biconnected:
        node_id, old_color, new_color = proposal

        return state.perimeter_computer.compactness_score(
            state.district_to_area[new_color],
            state.district_to_area[old_color],
            state.graph.nodes()[node_id]['area'],
            node_id,
            state.district_to_perimeter[old_color],
            state.district_to_perimeter[new_color],
            old_color, new_color, use_external_border = state.include_external_border)
    else:
        return _compactness_score(state, proposal)

def gml_score(state, proposal):

    from .neuro import gromov_wasserstein_barycenter_simple, gromov_wasserstein_discrepancy

    ot_hyperpara = {'loss_type': 'L2',  # the key hyperparameters of GW distance
               'ot_method': 'proximal',
             'alpha': 0,
               'beta': 2e-7,
               'outer_iteration': 300,
               # outer, inner iteration, error bound of optimal transport
               'iter_bound': 1e-4,
               'inner_iteration': 1,
               'sk_bound': 1e-4,
               'node_prior': 0,
               'max_iter': 200,  # iteration and error bound for calcuating barycenter
               'cost_bound': 1e-4,
               'update_p': False,  # optional updates of source distribution
               'lr': 0,
}




    # first - get new matrix mapping
    # {type: {graph_id: adjacency_matrix }}
    matrix_lookup = get_matrix_update(state, proposal)
    state._updated_matrix_lookup = matrix_lookup
    new_barycenter_lookup = {}

    for group_id, adj_mat_lookup in matrix_lookup.items():
        old_barycenter = state.barycenter_lookup[group_id]
        barycenter, _, _ = gromov_wasserstein_barycenter_simple(old_barycenter, adj_mat_lookup,
                                                                bary_size=state.parcellation_matrix.shape[1], ot_hyperpara=ot_hyperpara)
        new_barycenter_lookup[group_id] = barycenter
    discrepancy_sum = 0

    state._barycenter_proposal = new_barycenter_lookup # load this for future use if we accept the proposal

    for group_id, bary in new_barycenter_lookup.items():
        for other_group_id, other_bary in new_barycenter_lookup.items():
            if group_id > other_group_id:
                p_s = p_t = np.ones(shape=(bary.shape[0],1))
                discrepancy_sum += gromov_wasserstein_discrepancy(bary, other_bary, p_s, p_t, ot_hyperpara=ot_hyperpara)[1]

    return discrepancy_sum

# def update_matrix(state):
#
#     if not hasattr(state, 'matrix_updated'):
#         state.matrix_updated = state.iteration # make sure this is right
#
#
#     for move in state.move_log[state.matrix_updated:]:
#         if move is not None:
#             node_id, old_color, new_color = move
#
#             state.parcellation_matrix[node_id, old_color] -= 1
#             state.parcellation_matrix[node_id, new_color] += 1
#             state.matrix_lookup = state._updated_matrix_lookup



def eigen_score_inner(eigen_lookup, alpha=1.2, p=2):
    # TODO hardcoding here - should extract groups automatically
    return ((np.exp(alpha*eigen_lookup[-1])-np.exp(alpha*eigen_lookup[1]))**p).sum()
    # np.exp(alpha*(eigen_lookup[-1]-eigen))



def eigen_score(state, proposal):

    matrices_new = get_matrix_update(state,proposal)
    eigen_lookup_new = eigen_naive(matrices_new)
    return eigen_score_inner(eigen_lookup_new)-eigen_score_inner(state.eigen_lookup)

def frobenius_score(state, proposal):
    pass

    # this one is linear, yes?


def _compactness_score(state, proposal):
    prop_node, old_color, new_color = proposal # unpack

    area_prop = state.graph.nodes()[prop_node]['area'] if 'area' in state.graph.nodes()[prop_node] else 1

    perim_smaller = state.district_to_perimeter[old_color]
    area_smaller = state.district_to_area[old_color]

    perim_larger = state.district_to_perimeter[new_color]
    area_larger = state.district_to_area[new_color]

    # node_neighbors = state.graph.neighbors(prop_node)

    perim_larger_new, perim_smaller_new = perim_larger, perim_smaller

    for neighbor in state.graph.neighbors(prop_node):
        border_length = state.graph.edges[(prop_node, neighbor)]['border_length']
        if neighbor in state.color_to_node[new_color]:
            # we need to reduce the perimeter of new_color by their shared amount
            perim_larger_new -= border_length
            perim_smaller_new -= border_length

        elif neighbor in state.color_to_node[old_color]:
            # we need to increase the perimeter of old_color by their shared amount
            perim_larger_new += border_length
            perim_smaller_new += border_length

        else:
            # we need to increase the perimeter of new_color AND decrease of old color. no change to the perimeter of the 3rd district
            perim_larger_new += border_length
            perim_smaller_new -= border_length

    # perim_larger_new = perim_larger + sum([state.graph.edges()[(prop_node, other_node)]['border_length'] for other_node in node_neighbors
    #                                        if other_node not in state.color_to_node[new_color]])
    # perim_smaller_new = perim_smaller - sum([state.graph.edges()[(prop_node, other_node)]['border_length'] for other_node in node_neighbors
    #                                        if other_node not in state.color_to_node[old_color]])

    if state.include_external_border:
        perim_larger_new += state.graph.nodes()[prop_node]['external_border']
        perim_smaller_new -= state.graph.nodes()[prop_node]['external_border']

    score_old = perim_smaller**2/area_smaller + perim_larger**2/area_larger
    score_new = perim_smaller_new**2/(area_smaller-area_prop) + perim_larger_new**2/(area_larger+area_prop)

    # return score_old - score_new # the delta
    return score_new-score_old


def cut_length_score(state, proposal):
    # delta in cut length
    #node id, old_color, new_color
    neighbors = set(state.graph.neighbors(proposal[0]))
    # return len([i for i in neighbors if i in state.color_to_node[proposal[2]]]) - len([i for i in neighbors if i in state.color_to_node[proposal[1]]])
    score = (len(neighbors.intersection(state.color_to_node[proposal[1]])) - len(neighbors.intersection(state.color_to_node[proposal[2]])))
    # score = len(neighbors.intersection(state.color_to_node[proposal[2]])) - len(neighbors.intersection(state.color_to_node[proposal[1]]))
    return score




# this computes new population balance score, NOT the delta
def population_balance_score(state, proposal):
    node_id, old_color, new_color = proposal
    flipped_pop = state.graph.nodes()[node_id]['population']
    score_delta = 2*flipped_pop*(state.population_counter[new_color] - state.population_counter[old_color] + flipped_pop)
    return np.sqrt(state.population_deviation**2 + score_delta)


def population_balance_sq_score(state, proposal):
    node_id, old_color, new_color = proposal
    flipped_pop = state.graph.nodes()[node_id]['population']
    return 2 * flipped_pop * (state.population_counter[new_color] - state.population_counter[old_color] + flipped_pop)

