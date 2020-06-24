import random
import numpy as np
import collections


def state_log_to_coloring(process):
    '''converts the state log to a an array of colorings, structured . only functions if coloring is an int'''
    import numpy as np
    # transform an initial
    initial_coloring = process._initial_state.node_to_color
    # nodes = set(initial_coloring.keys())
    # node_list = sorted(initial_coloring.keys()) # ensure consistent
    node_list = list(process.state.graph.nodes())
    node_idx_lookup = {node: node_list.index(node) for node in node_list}
    coloring_array = np.ndarray(shape=(len(process.state.state_log), len(process.state.graph.nodes())),
                                dtype='int64')

    coloring_array[0, :] = [initial_coloring[i] for i in node_list]
    for i in range(1, len(process.state.state_log)):
        node_id, old_color, new_color = process.state.state_log[i]  # this might be skipping the first move?
        prev_state = coloring_array[(i - 1), :]  # check that this returns a copy not a pointer

        prev_state[node_idx_lookup[node_id]] = new_color
        coloring_array[i, :] = prev_state

    return coloring_array



def compute_autocorr_bootstrap(process, points = 10000, max_distance=500000):


    # pick points randomly

    move_loc = random.sample(list(range(process.move_log)), points)

    # num_colors = len(process.state.color_to_node)
    state_array = np.zeros(shape=(len(process.state.node_to_color), len(process.state.color_to_node)))
    node_to_idx = {b:a for a,b in enumerate(process.state.node_to_color.keys())}

    move_loc_to_state = {idx: None for idx in move_loc}


    for node, color in process._initial_state.node_to_color.items():
        state_array[node_to_idx[node], color] = 1
        # TODO assumption - colors are zero-indexed integers - can we do this?

    for i in range(len(process.move_log)):
        if process.move_log[i] is not None:
            # handle the move
            node_id, old_color, new_color = process.move_log[i]
            state_array[node_to_idx[node], old_color] = 0
            state_array[node_to_idx[node], new_color] = 1

        if i in move_loc_to_state:
            move_loc_to_state[i] = state_array.copy()

    # subtract off mu
    mu = np.full(shape=state_array.shape)
    for move_loc in move_loc_to_state:
        move_loc_to_state[move_loc] -= mu


    diff = np.zeros(shape=max_distance)
    weights = diff.copy()
    for loc, ary in move_loc_to_state.items():
        for other_loc, other_ary in move_loc_to_state.items():
            # sorry for nested loop, probably avoidable with sorting
            dist = other_loc - loc
            if dist > 0 and dist < max_distance:
                stat = sum(np.matmul(ary.T, other_ary)) #

                diff[dist] = (diff[dist]*weights[dist] + stat)/(weights[dist]+1)
                weights[dist] = weights + 1

    return diff, weights

def count_node_colorings(process):

    # current_state = copy.deepcopy(process._initial_state)

    node_current_color = copy.deepcopy(process._initial_state.node_to_color)

    time_last_flipped = {node_id: 0 for node_id in node_current_color.keys()}
    zeros = {color: 0 for color in process.state.color_to_node.keys()}
    node_coloring = {node_id: copy.deepcopy(zeros) for node_id in node_current_color.keys()}

    for i in range(len(process.state.move_log)):
        move = process.state.move_log[i] # (node_id, old_color, new_color
        if move is None:
            continue # should we actually continue or bump time_last_flipped?

        node_coloring[move[0]][move[1]] += i - time_last_flipped[move[0]] #
        time_last_flipped[move[0]] = i
        node_current_color[move[0]] = move[2]

    for node in node_coloring.keys():
        node_coloring[node][node_current_color[node]] += len(process.state.move_log) - time_last_flipped[node]

    return node_coloring

def count_node_flips(process):

    # count the number of times each node was flipped
    return collections.Counter([i[0] for i in process.state.move_log if i is not None])

