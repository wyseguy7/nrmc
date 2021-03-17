import numpy as np
try:
    from .biconnected import biconnected_dfs, dot_product, calculate_com_inner, PerimeterComputer
    cython_biconnected = True
except ImportError:
    cython_biconnected = False


def get_matrix_update(state, proposal):
    node_id, old_color, new_color = proposal
    neighbors = state.graph.neighbors(node_id)
    neighbor_vec = np.zeros(shape=(state.p, 1))

    for neighbor in neighbors:
        if state.node_to_color[neighbor] == old_color:
            neighbor_vec[neighbor] = 0
        elif state.node_to_color[neighbor] == new_color:
                neighbor_vec[neighbor] = -1


    inv_update_scalar_1 = 1 + neighbor_vec.T @ state.inv[node_id,:]

    inv_update_partial = state.inv - state.inv[node_id,:, None] @ neighbor_vec.T @ state.inv/inv_update_scalar_1
    # this is the Sherman-Morrison updated inverse for the first direction

    inv_update_scalar_2 = 1 + inv_update_partial[:, node_id, None].T @ neighbor_vec
    inv_update_full = inv_update_partial - inv_update_partial @  neighbor_vec @ inv_update_partial[None, :, node_id]/inv_update_scalar_2
    return inv_update_full, inv_update_scalar_1*inv_update_scalar_1


def car_model_score(state, proposal):

    # find updated inverse (XTX + Lambda) ^{-1}
    # inv_update_scalar =
    inv_update_full, inv_update_scalar = get_matrix_update(state, proposal)
    delta_likelihood = state.xty.T @ (state.inv - inv_update_full) @ state.xty
    score = state.phi*(delta_likelihood)*np.log(inv_update_scalar)

    state.prop_inv = inv_update_full # track this so we don't have to recompute
    state.prop_likelihood = state.likelihood + delta_likelihood
    return score

def car_model_updated(state, proposal):

    W_cop = state.W.copy()
    U_cop = state.U.copy()

    node_id, old_color, new_color = proposal
    neighbors = state.graph.neighbors(node_id)
    # neighbor_vec = np.zeros(shape=(state.p, 1))
    # W_cop[node_id, node_id] = 0 # reset this completely

    for neighbor in neighbors:
        if state.node_to_color[neighbor] == old_color:
            # each one loses a neighbor
            W_cop[node_id, node_id] -= 1
            W_cop[neighbor, neighbor] -= 1

            W_cop[node_id, neighbor] = 0
            W_cop[neighbor, node_id] = 0
        elif state.node_to_color[neighbor] == new_color:
            W_cop[node_id, neighbor] = -1
            W_cop[neighbor, node_id] = -1

            # each one gains a neighbor
            W_cop[node_id, node_id] += 1
            W_cop[neighbor, neighbor] += 1

    U_cop[node_id, old_color] = 0
    U_cop[node_id, new_color] = 1

    Lambda = state.get_lambda(W_cop)
    lu = Lambda @ U_cop
    inner = np.linalg.inv(U_cop.T @ lu + np.identity(U_cop.shape[1]))
    Phi = Lambda - lu @ inner @ lu.T
    inv_updated = np.linalg.inv(state.xtx + Phi)

    prop_det_log = np.linalg.slogdet(inv_updated)[1] - np.linalg.slogdet(Lambda)[1] - np.linalg.slogdet(inner)[1]

    delta_likelihood = (state.xty.T @ (inv_updated - state.inv) @ state.xty)
    state.prop_likelihood = state.likelihood + delta_likelihood
    state.prop_inv = inv_updated
    state.prop_W = W_cop
    state.prop_det_log = prop_det_log
    state.prop_U = U_cop

    return -1*((state.phi*delta_likelihood )+0.5*(prop_det_log-state.inv_det_log))


def car_model_score_naive(state, proposal):

    W_cop = state.W.copy()

    node_id, old_color, new_color = proposal
    neighbors = state.graph.neighbors(node_id)
    # neighbor_vec = np.zeros(shape=(state.p, 1))
    # W_cop[node_id, node_id] = 0 # reset this completely

    for neighbor in neighbors:
        if state.node_to_color[neighbor] == old_color:
            # each one loses a neighbor
            W_cop[node_id, node_id] -= 1
            W_cop[neighbor, neighbor] -= 1

            W_cop[node_id, neighbor] = 0
            W_cop[neighbor, node_id] = 0
        elif state.node_to_color[neighbor] == new_color:
            W_cop[node_id, neighbor] = -1
            W_cop[neighbor, node_id] = -1

            W_cop[node_id, node_id] += 1
            W_cop[neighbor, neighbor] += 1

            # W_cop[node_id, neighbor] = 1
            # W_cop[neighbor, node_id] = 1

    # inv_updated = np.linalg.inv(state.xtx + state.rho*W_cop + (1-state.rho)*np.identity(state.W.shape[0])+state.P1)
    inv_updated = state.get_inv(W_cop, state.phi)


    lda_prop = state.get_lambda(W_cop)
    # lda_current = state.get_lambda(state.W)

    prop_det_log = np.linalg.slogdet(inv_updated)[1] - np.linalg.slogdet(lda_prop)[1]

    delta_likelihood = (state.xty.T @ (inv_updated - state.inv) @ state.xty)
    state.prop_likelihood = state.likelihood + delta_likelihood
    state.prop_inv = inv_updated
    state.prop_W = W_cop
    state.prop_det_log = prop_det_log

    return -1*((state.phi*delta_likelihood )+0.5*(prop_det_log-state.inv_det_log))
    # np.linalg.inv(state.xtx + rho*state.W + (1-rho)*np.identity(state.W.shape[0]))

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

