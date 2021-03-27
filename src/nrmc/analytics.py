import random
import numpy as np
import collections
import copy
from collections import defaultdict

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

def compute_autocorr_new(process, points=1000, max_distance=5000000, intervals=200):
    if len(process.state.move_log) < max_distance:
        raise ValueError("Must specify max_distance lower than the sampling run")

    move_loc = random.sample(list(range(len(process.state.move_log) - max_distance)), points)

    interval_fixed = np.arange(0, max_distance, step=int(max_distance/intervals))
    # interval_loc = move_loc +

    # num_colors = len(process.state.color_to_node)
    state_array = np.zeros(shape=(len(process.state.node_to_color), len(process.state.color_to_node)))
    node_to_idx = {b: a for a, b in enumerate(process.state.node_to_color.keys())}

    move_loc_to_state = {idx: None for idx in move_loc}

    for node, color in process._initial_state.node_to_color.items():
        state_array[node_to_idx[node], color] = 1
        # TODO assumption - colors are zero-indexed integers - can we do this?

    for i in range(len(process.state.move_log)):
        if process.state.move_log[i] is not None:
            # handle the move
            node_id, old_color, new_color = process.state.move_log[i]
            state_array[node_to_idx[node_id], old_color] = 0
            state_array[node_to_idx[node_id], new_color] = 1

        if i in move_loc_to_state:
            # move_loc_to_state[i] = state_array.copy()
            move_loc_to_state[i] = copy.deepcopy(state_array.copy())


    iteration_to_interval = collections.defaultdict(list)

    for loc in move_loc:
        for dist in interval_fixed:
            iteration_to_interval[loc+dist].append((loc, dist))

    # subtract off mu from each array
    mu = np.full(shape=state_array.shape, fill_value=1 / len(process.state.color_to_node))
    for move_loc in move_loc_to_state:
        move_loc_to_state[move_loc] -= mu

    # reset state array
    state_array = np.zeros(shape=(len(process.state.node_to_color), len(process.state.color_to_node)))
    for node, color in process._initial_state.node_to_color.items():
        state_array[node_to_idx[node], color] = 1
        # TODO assumption - colors are zero-indexed integers - can we do this?

    records = collections.defaultdict(dict)

    for i in range(len(process.state.move_log)):
        if process.state.move_log[i] is not None:
            # handle the move
            node_id, old_color, new_color = process.state.move_log[i]
            state_array[node_to_idx[node_id], old_color] = 0
            state_array[node_to_idx[node_id], new_color] = 1

        adj_state = (state_array - mu).T

        for loc, dist in iteration_to_interval[i]:
            stat = np.trace(np.matmul(adj_state, move_loc_to_state[loc]))
            records[loc][dist] = stat


    return records


def find_corner_moves(process):

    corner_moves = list()
    state_array = np.zeros(shape=(len(process.state.node_to_color), len(process.state.color_to_node)))
    node_to_idx = {b: a for a, b in enumerate(process.state.node_to_color.keys())}

    corner_nodes = [0, 40, 1640, 1680] # SW, SE, NW, NE corners respectively

    sw_idx = node_to_idx[0]
    se_idx = node_to_idx[40]
    nw_idx = node_to_idx[1640]
    ne_idx = node_to_idx[1680]

    def corner_ownership(state_array):
        # node that this function will return None if corner ownership is not well-defined, which happens frequently
        if state_array[sw_idx][0]==state_array[nw_idx][0]:
            if state_array[se_idx][0] == state_array[ne_idx][0]:
                return 0 + 2*state_array[sw_idx][0]
        elif state_array[sw_idx][0] == state_array[se_idx][0]:
                if state_array[nw_idx][0] == state_array[ne_idx][0]:
                    return 1 + 2*state_array[sw_idx][0]

    for node, color in process._initial_state.node_to_color.items():
        state_array[node_to_idx[node], color] = 1
        # TODO assumption - colors are zero-indexed integers - can we do this?

    response = []
    for i in range(len(process.state.move_log)):
        if process.state.move_log[i] is not None:
            # handle the move
            node_id, old_color, new_color = process.state.move_log[i]
            state_array[node_to_idx[node_id], old_color] = 0
            state_array[node_to_idx[node_id], new_color] = 1

        response.append(corner_ownership(state_array))

    return response


def compute_autocorr_fixed_points(process, points=1000, max_distance=500000):
    if len(process.state.move_log) < max_distance:
        raise ValueError("Must specify max_distance lower than the sampling run")

    move_loc = random.sample(list(range(len(process.state.move_log) - max_distance)), points)

    # num_colors = len(process.state.color_to_node)
    state_array = np.zeros(shape=(len(process.state.node_to_color), len(process.state.color_to_node)))
    node_to_idx = {b: a for a, b in enumerate(process.state.node_to_color.keys())}

    move_loc_to_state = {idx: None for idx in move_loc}

    for node, color in process._initial_state.node_to_color.items():
        state_array[node_to_idx[node], color] = 1
        # TODO assumption - colors are zero-indexed integers - can we do this?

    for i in range(len(process.state.move_log)):
        if process.state.move_log[i] is not None:
            # handle the move
            node_id, old_color, new_color = process.state.move_log[i]
            state_array[node_to_idx[node_id], old_color] = 0
            state_array[node_to_idx[node_id], new_color] = 1

        if i in move_loc_to_state:
            # move_loc_to_state[i] = state_array.copy()
            move_loc_to_state[i] = copy.deepcopy(state_array.copy())

    # subtract off mu
    mu = np.full(shape=state_array.shape, fill_value=1 / len(process.state.color_to_node))
    for move_loc in move_loc_to_state:
        move_loc_to_state[move_loc] -= mu

    # reset state array to initial state TODO This is lazy
    for node, color in process._initial_state.node_to_color.items():
        state_array[node_to_idx[node], color] = 1

    move_loc_sorted = [(loc, move_loc_to_state[loc]) for loc in sorted(move_loc_to_state.keys())]

    diff = np.zeros(shape=max_distance)
    weights = diff.copy()

    start_idx = 0

    for i in range(len(process.state.move_log)):
        if process.state.move_log[i] is not None:
            # handle the move
            node_id, old_color, new_color = process.state.move_log[i]
            state_array[node_to_idx[node_id], old_color] = 0
            state_array[node_to_idx[node_id], new_color] = 1


        while i-move_loc_sorted[start_idx][0] >= max_distance:
            start_idx += 1 # increase start_idx until we are starting within max_distance

        for other_loc, other_array in move_loc_sorted[start_idx:]:

            if other_loc > i:
                break # we are now past i

            dist = i-other_loc

            stat = np.trace(np.matmul((state_array-mu).T, other_array))

            # diff[dist] = (diff[dist]*weights[dist] + stat)/(weights[dist]+1)
            diff[dist] += stat
            weights[dist] += 1

    return diff, weights


def compute_autocorr_bootstrap(process, points = 10000, max_distance=500000):

    if len(process.state.move_log) < max_distance:
        raise ValueError("Must specify max_distance lower than the sampling run")

    move_loc = random.sample(list(range(len(process.state.move_log))), points)

    # num_colors = len(process.state.color_to_node)
    state_array = np.zeros(shape=(len(process.state.node_to_color), len(process.state.color_to_node)))
    node_to_idx = {b: a for a, b in enumerate(process.state.node_to_color.keys())}

    move_loc_to_state = {idx: None for idx in move_loc}

    for node, color in process._initial_state.node_to_color.items():
        state_array[node_to_idx[node], color] = 1
        # TODO assumption - colors are zero-indexed integers - can we do this?

    for i in range(len(process.state.move_log)):
        if process.state.move_log[i] is not None:
            # handle the move
            node_id, old_color, new_color = process.state.move_log[i]
            state_array[node_to_idx[node_id], old_color] = 0
            state_array[node_to_idx[node_id], new_color] = 1

        if i in move_loc_to_state:
            # move_loc_to_state[i] = state_array.copy()
            move_loc_to_state[i] = copy.deepcopy(state_array.copy())

    # subtract off mu
    mu = np.full(shape=state_array.shape, fill_value=1 / len(process.state.color_to_node))
    for move_loc in move_loc_to_state:
        move_loc_to_state[move_loc] -= mu

    diff = np.zeros(shape=max_distance)
    weights = diff.copy()

    move_loc_sorted = [(loc, move_loc_to_state[loc]) for loc in sorted(move_loc_to_state.keys())]

    for i in range(len(move_loc_sorted)):

        loc, ary = move_loc_sorted[i]

        for other_loc, other_ary in iter(move_loc_sorted[i:]):
            # sorry for nested loop, probably avoidable with sorting
            dist = other_loc - loc
            if dist >= max_distance:
                break

            # stat = (ary==other_ary).sum()
            stat = np.trace(np.matmul(ary.T, other_ary))

            # diff[dist] = (diff[dist]*weights[dist] + stat)/(weights[dist]+1)
            diff[dist] += stat
            weights[dist] += 1

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


def track_state(process, to_track, updaters=()):
    # replay the move log and keep track of each attribute
    state = copy.deepcopy(process._initial_state) # sorry for inefficiency, but this will be miserable otherwise
    tracker = collections.defaultdict(list)

    # to_track is dict[str:function]

    for updater in updaters:
        updater(state)

    for move in process.state.move_log: # the actual moves
        state.handle_move(move)

        for updater in updaters:
            updater(state) # keep everything up to date

        for attr in to_track:
            tracker[attr].append(to_track[attr](state))

    return tracker # all done

def compute_districts_won(process, vote_map):
    # vote map is {node_id: (float, float)}

    # count the number of districts that 1st party wins

    district_to_votes = {district_id: np.array([0,0]) for district_id in process.state.color_to_node.keys()}
    for node_id, (p1, p2) in vote_map.items():
        district_id = process._initial_state.node_to_color[node_id]
        district_to_votes[district_id] += np.array([p1, p2])

    district_won = []
    for move in process.state.move_log:

        if move is not None:
            # update district_to_votes
            node_id, old_color, new_color = move
            p1, p2 = vote_map[node_id]
            district_to_votes[old_color] -= np.array([p1, p2])
            district_to_votes[new_color] += np.array([p1, p2])

        # count number of districts won
        wins = sum(v[0] > v[1] for v in district_to_votes.values())
        district_won.append(wins)

    return district_won


def compute_marginals(process, vote_map):

    district_to_votes = {district_id: np.array([0,0]) for district_id in process.state.color_to_node.keys()}
    for node_id, (p1, p2) in vote_map.items():
        district_id = process._initial_state.node_to_color[node_id]
        district_to_votes[district_id] += np.array([p1, p2])

    marginal_list = []
    for move in process.state.move_log:

        if move is not None:

            node_id, old_color, new_color = move
            p1, p2 = vote_map[node_id]
            district_to_votes[old_color] -= np.array([p1, p2])
            district_to_votes[new_color] += np.array([p1, p2])

        marginals = sorted([i[1]-i[0] for i in district_to_votes.values()])
        marginal_list.append(copy.deepcopy(marginals))

    return marginal_list


def rolling_weighted_mean(sample, times, weights, window_size=1000):


    half_window_size = window_size/2
    start_idx = 0
    end_idx = int(half_window_size)

    for i in range(times[0], times[-1]):

        while i- times[start_idx] > half_window_size:
            start_idx += 1

        end_idx += 1
        while times[end_idx] - i  > half_window_size:
            end_idx -= 1


def coreset_transitions(process, interval=0.5):


    state_array = np.zeros(shape=(len(process.state.node_to_color), len(process.state.color_to_node)))
    node_to_idx = {b: a for a, b in enumerate(process.state.node_to_color.keys())}

    for node, color in process._initial_state.node_to_color.items():
        state_array[node_to_idx[node], color] = 1
        # TODO assumption - colors are zero-indexed integers - can we do this?


    # assumes a rectangular lattice
    centroids = {i: process.state.graph.nodes()[i]['Centroid'] for i in node_to_idx.keys()}
    x_coord = [i[0] for i in centroids.values()]
    y_coord = [i[1] for i in centroids.values()]
    bounding_box = ((min(x_coord), max(x_coord)), (min(y_coord), max(y_coord)))
    dimensions = bounding_box[0][1] - bounding_box[0][0], bounding_box[1][1] - bounding_box[1][0],

    midpoint = (
    (bounding_box[0][0] + bounding_box[0][1]) / 2, (bounding_box[1][0] + bounding_box[1][1]) / 2)  # x, y midpoint

    orientation_idx = 0
    other_idx = 1

    # find indices for horizontal orientation
    sw_idx, se_idx, ne_idx, nw_idx = None, None, None, None
    sw_ext, se_ext, nw_ext, ne_ext = bounding_box[1][0], bounding_box[1][0], bounding_box[1][1], bounding_box[1][
        1]  # change for vertical

    for idx, centroid in centroids.items():

        if centroid[orientation_idx] == bounding_box[orientation_idx][0]:  # on western boundary
            if centroid[other_idx] > dimensions[other_idx] * interval / 2 + midpoint[other_idx] and centroid[
                other_idx] < nw_ext:
                nw_ext = centroid[other_idx]
                nw_idx = idx
            if centroid[other_idx] < midpoint[other_idx] - dimensions[other_idx] * interval / 2 and centroid[
                other_idx] > sw_ext:
                sw_ext = centroid[other_idx]
                sw_idx = idx

        elif centroid[orientation_idx] == bounding_box[orientation_idx][1]:  # on eastern boundary
            if centroid[other_idx] > dimensions[other_idx] * interval / 2 + midpoint[other_idx] and centroid[
                other_idx] < ne_ext:
                ne_ext = centroid[other_idx]
                ne_idx = idx
            if centroid[other_idx] < midpoint[other_idx] - dimensions[other_idx] * interval / 2 and centroid[
                other_idx] > se_ext:
                se_ext = centroid[other_idx]
                se_idx = idx

    horizontal_idx = [node_to_idx[node] for node in [sw_idx, se_idx, ne_idx, nw_idx]]

    # find indices for vertical orientation
    sw_idx, se_idx, ne_idx, nw_idx = None, None, None, None
    sw_ext, se_ext, nw_ext, ne_ext = bounding_box[0][0], bounding_box[0][1], bounding_box[0][0], bounding_box[0][1]

    orientation_idx = 1
    other_idx = 0

    for idx, centroid in centroids.items():

        if centroid[orientation_idx] == bounding_box[orientation_idx][0]:  # on southern boundary
            if centroid[other_idx] > dimensions[other_idx] * interval / 2 + midpoint[other_idx] and centroid[
                other_idx] < se_ext:
                se_ext = centroid[other_idx]
                se_idx = idx
            if centroid[other_idx] < midpoint[other_idx] - dimensions[other_idx] * interval / 2 and centroid[
                other_idx] > sw_ext:
                sw_ext = centroid[other_idx]
                sw_idx = idx

        elif centroid[orientation_idx] == bounding_box[orientation_idx][1]:  # on northern boundary
            if centroid[other_idx] > dimensions[other_idx] * interval / 2 + midpoint[other_idx] and centroid[
                other_idx] < ne_ext:
                ne_ext = centroid[other_idx]
                ne_idx = idx
            if centroid[other_idx] < midpoint[other_idx] - dimensions[other_idx] * interval / 2 and centroid[
                other_idx] > nw_ext:
                nw_ext = centroid[other_idx]
                nw_idx = idx

    vertical_idx = [node_to_idx[node] for node in [sw_idx, se_idx, ne_idx, nw_idx]]

    def corner_ownership(state_array):
        # sw_idx, se_idx, ne_idx, nw_idx

        if state_array[horizontal_idx[0]][0] != state_array[horizontal_idx[3]][0] and state_array[horizontal_idx[1]][0] != \
                state_array[horizontal_idx[2]][0]:
            # cut is horizontal
            return 0 + 2 * state_array[horizontal_idx[0]][0]

        # se corner doesn't match sw corner and ne corner doesn't match nw corner
        elif state_array[vertical_idx[0]][0] != state_array[vertical_idx[1]][0] and state_array[vertical_idx[2]][0] != \
                state_array[vertical_idx[3]][0]:
            return 1 + 2 * state_array[vertical_idx[0]][0]


    response = []
    for i in range(len(process.state.move_log)):
        if process.state.move_log[i] is not None:
            # handle the move
            node_id, old_color, new_color = process.state.move_log[i]
            state_array[node_to_idx[node_id], old_color] = 0
            state_array[node_to_idx[node_id], new_color] = 1

        response.append(corner_ownership(state_array))


    return response


def extract_center_of_mass(process):
    from nrmc.updaters import update_center_of_mass

    process = copy.deepcopy(process) # just to be safe

    com_centroids_log = []

    for i in range(len(process.state.move_log)):
        update_center_of_mass(process._initial_state)

        com_centroids_log.append(copy.deepcopy(process._initial_state.com_centroid))

        process._initial_state.handle_move(process.state.move_log[i])
        # process._initial_state.iteration+=1

    my_dict = {}
    for district_id in com_centroids_log[0]:
        output = np.ndarray(shape=(len(com_centroids_log), 2))

        for i in range(len(com_centroids_log)):
            output[i,:] = com_centroids_log[i][district_id]
        my_dict[district_id] = output

    return my_dict



def center_of_mass_to_polar(com_centroids, center):

    com_log_polar = {}
    for district_id in com_centroids:
        # convert to polar
        theta = np.arctan2(com_centroids[district_id][:,1]-center[1], com_centroids[district_id][:,0]-center[0])
        com_log_polar[district_id] = theta
    com_polar = {}

    for district_id in com_log_polar:

        com = com_log_polar[district_id]
        diff = np.ndarray(shape=(com.shape[0], 1))
        diff[0] = com[0]

        for i in range(1, com.shape[0]):
            if com[i-1]*com[i] < 0 and np.abs(com[i]-com[i-1])>np.pi:
                # differing signs
                if com[i] > 0 and com[i-1] < 0:
                    diff[i] = com[i]-com[i-1] - 2*np.pi
                else:
                    # com[i] < 0 and com[i-1] > 0
                    diff[i] = com[i]-com[i-1] + 2*np.pi
            else:
                diff[i] = com[i]-com[i-1]



        output = np.cumsum(diff)
        com_polar[district_id] = output
        # com_polar[district_id] = diff # TODO fix


    return com_polar

    #
    # for i in range(1, len(com_log_polar)):
    #     my_dict = {}
    #     for district_id in com_log_polar[i]:
    #         if com_log_polar[i-1][district_id]*com_log_polar[i][district_id] < 0:
    #             pass
    #         else:
    #             my_dict[district_id] =
    #
    #
    #
    #
    #
    #
    # for i in range(len(process.state.move_log)):
    #     update_center_of_mass(process.state)
    #     for district_id in process.state.color_to_node:
    #         rot_matrix = get_rot_matrix(-com_state[district_id])
    #         new_points = np.matmul(rot_matrix, process.state.com_centroid[district_id]-center) # rotated points
    #         delta_theta = np.arctan2(new_points[0, 0], new_points[0,1])
    #         com_log[district_id].append(com_log[district_id][-1]+delta_theta)
    #
    #         new_theta = np.arctan2(*(process.state.com_centroid[district_id]-center)) # calculate for next time
    #         com_state[district_id] = new_theta
    #
    #     process.state.iteration+=1
    #
    # return com_log


def get_rot_matrix(theta):
    return np.matrix([[ np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])


def state_to_apd(state_array, node_to_idx, node_to_population, ideal_pop=None):

    if ideal_pop is None:
        ideal_pop = sum(node_to_population.values())/state_array.shape[1]

    district_to_pop = defaultdict(float)
    for node_id, idx in node_to_idx.items():
        for j in range(state_array.shape[1]):
            if state_array[idx, j] == 1:

                district_to_pop[j] += node_to_population[node_id]
                break

    return max([abs(pop-ideal_pop) for pop in district_to_pop.values()])/ideal_pop

def calculate_apd(process, ideal_pop=None):

    apd_list = []

    state_array = np.zeros(shape=(len(process.state.node_to_color), len(process.state.color_to_node)))
    node_to_idx = {b: a for a, b in enumerate(process.state.node_to_color.keys())}
    node_to_population = {node_id: process.state.graph.nodes()[node_id]['population'] for node_id in process.state.graph.nodes()}

    # calculate initial apd
    for node, color in process._initial_state.node_to_color.items():
        state_array[node_to_idx[node], color] = 1
        # TODO assumption - colors are zero-indexed integers - can we do this?

    apd = state_to_apd(state_array, node_to_idx, node_to_population)

    for move in process.state.move_log:
        if move is not None:
            # update state
            node_id, old_color, new_color = move
            state_array[node_to_idx[node_id], old_color] = 0
            state_array[node_to_idx[node_id], new_color] = 1

            # recalculate
            apd = state_to_apd(state_array, node_to_idx, node_to_population, ideal_pop)
        apd_list.append(apd)

    return apd_list


def tv_dist_discrete(X, MAX_X=5):
    from collections import defaultdict
    # X is n times p numpy array
    n, p = X.shape
    # counters = [defaultdict(int) for _ in range(p)]
    counters = [{k: 0 for k in range(MAX_X + 1)} for _ in range(p)]
    tv = []

    for i in range(n):
        for j in range(p):
            counters[j][X[i, j]] += 1

        max_diff = 0.
        for k in range(MAX_X + 1):
            max_k = max(counter[k] for counter in counters)
            min_k = min(counter[k] for counter in counters)  # forgive me for repeating these loops
            max_diff = max(max_diff, max_k - min_k)
        tv.append(max_diff)

    return [tv[i] / (i + 1) for i in range(tv)]  # don't forget to normalize!