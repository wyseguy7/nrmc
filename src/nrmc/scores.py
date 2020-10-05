import numpy as np


def compactness_score(state, proposal):
    prop_node, old_color, new_color = proposal # unpack
    score, score_prop = 0, 0

    area_prop = state.graph.nodes()[prop_node]['area'] if 'area' in state.graph.nodes()[prop_node] else 1

    perim_smaller = state.district_to_perimeter[old_color]
    area_smaller = state.district_to_area[old_color]

    perim_larger = state.district_to_perimeter[new_color]
    area_larger = state.district_to_area[new_color]

    # node_neighbors = state.graph.neighbors(prop_node)

    perim_larger_new, perim_smaller_new = perim_larger, perim_smaller

    for neighbor in state.graph.neighbors(prop_node):
        if neighbor in state.color_to_node[new_color]:
            # we need to reduce the perimeter of new_color by their shared amount
            perim_larger_new -= state.graph.edges[(prop_node, neighbor)]['border_length']

        elif neighbor in state.color_to_node[old_color]:
            # we need to increase the perimeter of old_color by their shared amount
            perim_smaller_new += state.graph.edges[(prop_node, neighbor)]['border_length']

        else:
            # we need to increase the perimeter of new_color AND decrease of old color. no change to the perimeter of the 3rd district
            perim_larger_new += state.graph.edges[(prop_node, neighbor)]['border_length']
            perim_smaller_new -= state.graph.edges[(prop_node, neighbor)]['border_length']

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

