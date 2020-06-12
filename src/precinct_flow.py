import uuid
import os
import pickle
import logging
import networkx as nx
from collections import namedtuple, defaultdict
import numpy as np
import copy
import random
import collections
import itertools

try:
    from biconnected import biconnected_dfs, dot_product
    cython_biconnected = True
except ImportError:
    print("No Cython for you!")
    cython_biconnected = False

# Remaining to-dos
# Sort out state initializer, object loader
# Details on COM flow algorithm


# BFS on neighbors of potentially disconnected


####### Meeting notes 2019-12-06:
# Make 'state' with listeners that track need for metadata

# state object that allows arbitrary extensions
ROT_MATRIX = np.matrix([[0, -1], [1, 0]])
CENTROID_DIM_LENGTH = 2 # TODO do we need a settings.py file?
exp = lambda x: np.exp(min(x, 700)) # avoid overflow


class State(object):

    def __init__(self, graph, coloring, tallied_stats=('node_count','population'), log_contested_edges = True,
                 coerce_int = True):


        if coerce_int:
            # this is important to ensure it's compatible with our Cython speedups
            relabeling = {i: int(i) for i in graph.nodes()}
            coloring = {int(k):v for k,v in coloring.items()}
            graph = nx.relabel_nodes(graph, relabeling, copy=True)

        self.coerce_int = coerce_int
        self.graph = graph

        self.node_to_color = coloring
        d = defaultdict(set)
        for node_id, district_id in coloring.items():
            d[district_id].add(node_id)
        self.color_to_node = dict(d)

        self.iteration = 0
        self.state_log = []
        self.move_log = [] # move log is state_log plus 'None' each time we make an involution

        self.stat_tally = defaultdict(dict)

        # for district_id, nodes in self.color_to_node.items():
        #     for stat in tallied_stats:
        #         self.stat_tally[stat][district_id] = sum(graph[node_id][stat] for node_id in nodes)
        #
        # self.tallied_stats = tallied_stats

        self.log_contested_edges = log_contested_edges # rip these out into a mixin?
        self.contested_edge_counter = collections.defaultdict(int)
        self.contested_node_counter = collections.defaultdict(int)

        self.check_connectedness = True

        self.connected_true_counter = [] # TODO remove after finished debugging
        self.connected_false_counter = []

        self.start_stop_to_chain = dict() # TODO remove
        self.node_to_start_stop = dict()




    def check_connected_lookup(self, proposal):
        # TODO finish this off so we can use it

        node_id, old_color, new_color = proposal

        # check that smaller one is still connected
        district_nodes = self.color_to_node[old_color].difference(
            {node_id})  # all of the old_color nodes without the one we're removing
        to_connect = district_nodes.intersection(
            set(self.graph.neighbors(node_id)))  # need to show each of these are still connected

        if not to_connect:  # TODO should this check get run here or should it be guaranteed?
            return False  # we deleted the last node of a district, this is illegal

        init = min(to_connect)
        to_connect = {i for i in to_connect if i!=init or (init, i) in self.start_stop_to_chain}

        to_search = [init]  # ordered list of nodes available to search
        visited_set = {init}

        connected_counter = 0

        while True:  # there are still nodes we haven't found
            connected_counter += 1

            # evaluate halting criteria
            if not to_connect - visited_set:  # we visited all of the nodes we needed to search
                self.connected_true_counter.append(connected_counter)
                return True
            if not to_search:  # no more nodes available to search
                # raise ValueError("")
                self.connected_false_counter.append(connected_counter)
                return False

            next_node = to_search.pop()

            # new nodes that we haven't seen yet
            new_neighbors = set(self.graph.neighbors(to_search.pop())).intersection(district_nodes).difference(
                visited_set)

            priority_neighbors = to_connect.intersection(new_neighbors)
            nonpriority_neighbors = new_neighbors - priority_neighbors
            visited_set.update(new_neighbors)  # seen new neighbors
            to_search.extend(nonpriority_neighbors)
            to_search.extend(priority_neighbors)  # ensure that priority neighbors are at the top of the queue

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def handle_move(self, move):
        self.move_log.append(move)
        self.iteration += 1 # always change this
        if move is not None:
            # self.state_log.append(move)
            self.flip(move[0], move[2]) # TODO check this indexing


    def flip(self, node_id, district_id):
        # TODO this should be an internal method, since if you start executing it it won't update _proposal_state as well
        # update lookups
        old_color = self.node_to_color[node_id]
        self.color_to_node[old_color].remove(node_id)
        self.color_to_node[district_id].add(node_id)
        self.node_to_color[node_id] = district_id

        # record the move


        # for stat in self.tallied_stats:
        #     # the stat tally giveth
        #     self.stat_tally[stat][district_id] += self.graph.nodes()[node_id][stat] # node_data[node_id][stat]
        #     # ...and it taketh away
        #     self.stat_tally[stat][old_color] -= self.graph.nodes()[node_id][stat] # self.node_data[node_id][stat]

    # shim to add in data

    @classmethod
    def from_file_hardcode(cls):
        # TODO update this for new graph stuff
        # just for testing purposes on Duplin Onslow
        filepath = 'nonreversiblecodebase/data/DuplinOnslow/Duplin_C_Onslow_P__BORDERLENGTHS.txt'
        # todo sort out

        edges = []
        node_data = dict()
        delim = '\t'
        border_key = '-1'
        with open(filepath) as f:
            for line in f:
                L = line.strip().split(delim)
                if L[0] != border_key and L[1] != border_key:  # ignore border for now
                    edges.append((int(L[0]), int(L[1])))

        population_filepath = 'nonreversiblecodebase/data/DuplinOnslow/Duplin_C_Onslow_P__POPULATION.txt'
        with open(population_filepath) as f:
            for line in f:
                L = line.strip().split(delim)
                node_id = int(L[0])
                population = float(L[1])
                node_data[node_id] = {'population': population, 'node_count': 1}  # instantiate with population

        centroids_filepath = 'nonreversiblecodebase/data/DuplinOnslow/Duplin_C_Onslow_P__CENTROIDS.txt'
        with open(centroids_filepath) as f:
            for line in f:
                L = line.strip().split(delim)
                node_id = int(L[0])
                node_data[node_id]['x'] = float(L[1])
                node_data[node_id]['y'] = float(L[2])

        # node to color - hardcoded to
        coloring = {0: 1001, 1: 1001, 10: 1001, 19: 1001, 20: 1001,
                    2: 1002, 12: 1002, 21: 1002, 22: 1002, 23: 1002,
                    3: 1003, 9: 1003, 7: 1003, 5: 1003, 6: 1003,
                    4: 1004, 14: 1004, 15: 1004, 16: 1004, 17: 1004, 24: 1004,
                    8: 1005,  # just leave this one as a single to see if that causes issues
                    11: 1006, 13: 1006, 18: 1006}

        node_list = list(coloring.keys())

        return State.from_edge_node_list(node_list, edges, coloring=coloring, node_data=node_data)


    @classmethod
    def from_edge_node_list(cls, node_list, edge_list, coloring, node_data=None, edge_data=None, tallied_stats=('population', 'node_count')):
        ''' allows user to initialize state as (V,E), as well as node_data, edge_data instead of passing a nx.graph object with info attached
        user must still provide coloring as before'''

        graph = nx.Graph()
        graph.add_nodes_from(node_list)
        graph.add_edges_from(edge_list)

        if node_data is not None:
            nx.set_node_attributes(graph, node_data)
        if edge_data is not None:
            nx.set_edge_attributes(graph, edge_data)

        return State(graph, coloring, tallied_stats=tallied_stats)

    @classmethod
    def from_state(cls, state):
        '''Shim that accepts 'legacy' state object and initializes new class '''
        coloring = state['nodeToDistrict']
        graph = state['graph'] # includes all necessary data? do we need to recode anything?

        return State(graph, coloring, tallied_stats=[]) # should tallied stats be different?


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


def naive_init_flow(node_dict, edge_list, centroid):
    # generator
    # accepts node_dict, edge_list, reoutputs in correct directed order u -> v
    # edge_list tuple u -> v
    # centroid is tuple (x,y) for flow. flow is always CCW .
    # TODO possibly allow for elliptical flow?

    for edge in edge_list:
        n1 = node_dict[edge[0]]  # the centerpoint between the two nodes
        n2 = node_dict[edge[1]]

        theta1 = np.math.atan2(n1.y - centroid[1], n1.x - centroid[0])
        theta2 = np.math.atan2(n2.y - centroid[1], n2.x - centroid[0])
        yield (edge if theta1 >= theta2 else (
            edge[1], edge[0]))  # TODO the logical is wrong, doesn't account for branch cut issue

def contested_edges_naive(state):
    # generate contested edges by testing each edge in the graph. it's brute force and definitely works
    contested = set()

    for edge in state.graph.edges:
        if state.node_to_color[edge[0]] != state.node_to_color[edge[1]]:
            contested.add((min(edge[0], edge[1]), max(edge[0], edge[1])))  # always store small, large
        # if state.graph.nodes[edge[0]]['coloring'] != graph.nodes[edge[1]]['coloring']:
        #     contested.add((min(edge), max(edge))) # always store small, large
    return contested

def update_articulation_points(state):

    if not hasattr(state, 'articulation_points'):

        ap_mapping = {}
        for district_id, nodes in state.color_to_node.items():
            ap_mapping[district_id] = set(nx.articulation_points(state.graph.subgraph(nodes)))

        state.articulation_points = ap_mapping

        # create arrays for all the things
        state.adj_mapping_full = {node: np.array(list(state.graph[node].keys()), dtype='i') for node in state.graph.nodes()}

        # create for just this one
        state.adj_mapping = {district_id: {node: np.array([j for j in state.graph[node] if state.node_to_color[j]==district_id], dtype='i')
                                           for node in nodes} for district_id, nodes in state.color_to_node.items()}

        state.articulation_points_updated = state.iteration

    updated_districts = set()
    for move in state.move_log[state.articulation_points_updated:]:
        if move is not None:
            node_id, old_color, new_color = move
            updated_districts.add(old_color)
            updated_districts.add(new_color)

            del state.adj_mapping[old_color][node_id] # remove from old
            # add to new
            state.adj_mapping[new_color][node_id] = np.array(
                [j for j in state.adj_mapping_full[node_id] if state.node_to_color[j]==new_color], dtype='i')

            for neighbor_id in state.adj_mapping_full[node_id]:
                neighbor_color = state.node_to_color[neighbor_id]
                state.adj_mapping[neighbor_color][neighbor_id] = np.array(
                    [i for i in state.adj_mapping_full[neighbor_id] if state.node_to_color[i]==neighbor_color], dtype='i')

            # state.adj_mapping[old_color][node_id] = np.array(
            #     [j for j in state.graph.nodes[node_id] if state.node_to_color[j]==old_color], dtype='i')
            # state.adj_mapping[old_color][node_id] = np.array(
            #     [j for j in state.graph.nodes[node_id] if state.node_to_color[j] == new_color], dtype='i')

    for updated_district in updated_districts:
        if cython_biconnected and state.coerce_int: # only use if we were able to import properly
            # provide node list and adjacency graph - is this going to double-pay for serialization?
            art_points = biconnected_dfs(list(state.color_to_node[updated_district]),
                                         state.adj_mapping[updated_district])
                                         # {i: [j for j in state.graph[i] if j in state.color_to_node[updated_district]]
                                          # for i in state.color_to_node[updated_district]})
        else:
            # TODO put a warning here that the user is using the slower algorithm
            art_points = nx.articulation_points(state.graph.subgraph(state.color_to_node[updated_district]))


        state.articulation_points[updated_district] = set(art_points)

    state.articulation_points_updated = state.iteration



# we can either subclass this object or attach a bunch of functions to it, either is fine really
class MetropolisProcess(object):

    def __init__(self, state, beta=1, measure_beta=1, minimum_population = None, log_com = True):
        self._initial_state = copy.deepcopy(state)  # save for later
        self.state = state
        self.score_log = [] # for debugging
        self.beta = beta
        self.measure_beta = measure_beta
        self.minimum_population = minimum_population
        if not hasattr(state, 'involution'):
            state.involution = 1 # unless this comes from elsewhere, it should have an involution

        self.log_com = log_com
        if log_com:
            self.com_log = []


    def make_sandbox(self):


        my_id = uuid.uuid4().hex[:6]
        self.filepath = "{classname}_{my_id}_measure_beta={measure_beta}beta={beta}".format(
            measure_beta=self.measure_beta, beta=self.beta, classname=self.__class__.__name__, my_id= my_id)
        os.makedirs(self.filepath, exist_ok=False) # ensure that the folder path exists

    def save(self):
        with open(os.path.join(self.filepath, 'process.pkl'), mode='wb') as f:
            pickle.dump(self)



    def score_to_prob(self, score):
        return exp(-0.5*self.measure_beta*score)

    def score_to_proposal_prob(self, score):
        return exp(-0.5*score*self.beta)

    def proposal_check(self, state, proposal):
        # hook to support subclassing
        return True


    def involve_state(self, state):
        state.involution *= -1

    def perform_involution(self):
        self.involve_state(self.state)

    def proposal_filter(self, state, proposals):

        update_boundary_nodes(state)
        update_articulation_points(state)
        pop_check_failed = set()

        for node_id, old_color, new_color in proposals:
            if old_color in pop_check_failed:
                continue

            if self.minimum_population is not None and not check_population(state, old_color, self.minimum_population):
                pop_check_failed.add(old_color)
                continue

            if node_id in state.articulation_points[old_color] or not simply_connected(state, node_id, old_color, new_color) \
                    or not self.proposal_check(state, (node_id, old_color, new_color)):
                continue # will disconnect the graph

            yield (node_id, old_color, new_color)


    def proposal_checks(self, state, proposal):
        # checked against each proposal in get_proposals
        node_id, new_color, old_color = proposal # should really make a named tuple for this, tired of unpacking

        if self.minimum_population is not None and not check_population(state, old_color, self.minimum_population):
            return False

        # if state.check_connectedness and not()
        # if state.check_connectedness and not ( connected_breadth_first(state, node_id, old_color)
        #                                        and simply_connected(state, node_id,old_color, new_color)):
        #     return False

        return True

    def score_proposal(self, node_id, old_color, new_color, state):
        return cut_length_score(state, (node_id, old_color, new_color))

    def get_directed_edges(self, state):
        # default behavior gets all edges in each direction
        update_contested_edges(state)
        return itertools.chain(state.contested_edges, [(edge[1], edge[0]) for edge in state.contested_edges])

    def get_proposals(self, state):
        # produces a mapping {proposal: score} for each edge in get_directed_edges

        scored_proposals = {}

        for node_id, neighbor in self.get_directed_edges(state):
            old_color = state.node_to_color[node_id]
            new_color = state.node_to_color[neighbor]

            if (node_id, old_color, new_color) in scored_proposals:
                continue # already scored this proposal

            # if not self.proposal_checks(state, (node_id, old_color, new_color)):
            #     continue # not a valid proposal

            scored_proposals[(node_id, old_color, new_color)] = self.score_proposal(node_id, old_color, new_color, state)

        return {proposal: scored_proposals[proposal] for proposal in self.proposal_filter(state, scored_proposals)}

        # return scored_proposals


    def proposal(self, state):
        # picks a proposal randomly without any weighting
        scored_proposals = self.get_proposals(self.state) # TODO check this against subclassing behaviour
        # score = self.score_proposal()
        proposal = random.choices(list(scored_proposals.keys()))[0]
        prob = self.score_to_prob(scored_proposals[proposal])  # should be totally unweighted here

        return proposal, prob

    def handle_acceptance(self, prop, state):
        state.handle_move(prop)

    def handle_rejection(self, prop, state):
        self.perform_involution()
        state.handle_move(None)

    def accept_reject(self, score):
        # classic Metropolis accept-reject. we can override if needed
        # TODO rng here
        # returns a boolean
        u = np.random.uniform()
        return u < score
        # return u < min(score, 1)

    def step(self):
        prop, prob = self.proposal(self.state)  # no side effects here, should be totally based on current state

        if self.accept_reject(prob):  # no side effects here
            # proposal accepted!

            self.handle_acceptance(prop, self.state)  # now side effects
        else:
            self.handle_rejection(prop, self.state)  # side effects also here

        if self.log_com:
            update_center_of_mass(self.state)
            self.com_log.append(self.state.com_centroid)

        self.score_log.append(len(self.state.contested_edges))  # TODO for debugging only


    def check_connected(self, state, node_id, old_color, new_color):
        return connected_breadth_first(state, node_id, old_color) and simply_connected(state, node_id, old_color, new_color)


    def check_connected_naive(self, state, node_id, old_color):
        # legacy method that uses networkx method

        # TODO change this to a set
        proposed_smaller = state.graph.subgraph(
            [i for i in state.color_to_node[old_color] if i != node_id])  # the graph that lost a node
        return len(proposed_smaller) and nx.is_connected(proposed_smaller)

class TemperedProposalMixin(MetropolisProcess):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.state.check_connectedness = False # turn this off - we check it differently here
        self._proposal_state = copy.deepcopy(self.state)

        # if involution is set, make it always the opposite
        self._proposal_state.involution = self.state.involution *-1 # set to opposite of original
        self._proposal_state.log_contested_edges = False # don't waste time logging this

        self.no_reverse_prob_counter = 0
        self.no_valid_proposal_counter = 0


    def perform_involution(self):
        self.involve_state(self.state)
        self.involve_state(self._proposal_state)

    def handle_acceptance(self, prop, state):

        super().handle_acceptance(prop, state)




    def handle_rejection(self, prop, state):
        super().handle_rejection(prop, state)
        if prop[0] is not None: # ignore if we couldn't find a valid proposal
            self._proposal_state.handle_move((prop[0], prop[2], prop[1]))  # undo earlier move

    def proposal(self, state):
        # TODO this is common with SingleNodeFlipTempered, perhaps split this out
        scored_proposals = self.get_proposals(self.state)
        proposal_probs = {k:self.score_to_proposal_prob(v) for k,v in scored_proposals.items()}

        try:
            proposal = random.choices(list(proposal_probs.keys()),
                                      weights=proposal_probs.values())[0]
        except IndexError:
            self.no_valid_proposal_counter += 1
            return (None, None, None), 0 # we couldn't find a valid proposal, need to involve


        self._proposal_state.handle_move(proposal)
        # if not self.check_connected(state, *proposal) and simply_connected(state, *proposal):
        #     return proposal, 0 # always zero

        reverse_proposals = self.get_proposals(self._proposal_state)
        reverse_proposal_probs = {k:self.score_to_proposal_prob(v) for k,v in reverse_proposals.items()}

        try:
            q = reverse_proposal_probs[(proposal[0], proposal[2], proposal[1])]/sum(reverse_proposal_probs.values())
            q_prime = proposal_probs[proposal]/sum(proposal_probs.values())
            prob = self.score_to_prob(scored_proposals[proposal])*q/q_prime
            return proposal, prob
        except KeyError:
            self.no_reverse_prob_counter += 1
            return proposal, 0 # this happens sometimes but probably shouldn't for single node flip


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


def calculate_com_one_step(state, proposal, weight_attribute=None):

    node_id, old_color, new_color = proposal

#     com_centroid = copy.deepcopy(state.com_centroid) # ugh, should this function just be side-effecting? how bad is this cost?
#     total_weight = copy.deepcopy(state.com_total_weight)
    node = state.graph.nodes()[node_id] # how expensive is this lookup, anyways?

    weight = node[weight_attribute] if weight_attribute is not None else 1


    centroid_new_color = (node['Centroid'] * weight + state.com_centroid[new_color] * state.com_total_weight[new_color])/(
            state.com_total_weight[new_color] + weight)
    centroid_old_color = (-node['Centroid'] * weight + state.com_centroid[old_color] * state.com_total_weight[old_color])/(
            state.com_total_weight[old_color] - weight)


    # for j in range(CENTROID_DIM_LENGTH):
    #     com_centroid[new_color][j] = (node[j] * weight + com_centroid[new_color][j] * total_weight[new_color]) / (
    #             total_weight[new_color] + weight)
    #     com_centroid[old_color][j] = (-node[j] * weight + com_centroid[old_color][j] * total_weight[old_color]) / (
    #             total_weight[old_color] - weight)
    # total_weight[new_color] = total_weight[new_color] + weight
    # total_weight[old_color] = total_weight[old_color] - weight

    total_weight_new_color = state.com_total_weight[new_color] + weight
    total_weight_old_color = state.com_total_weight[old_color] - weight

    return centroid_new_color, centroid_old_color, total_weight_new_color, total_weight_old_color

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



class PrecintFlow(MetropolisProcess):
    # TODO perhaps stat_tally should go in here instead

    # make proposals on what maximizes the L2 norm of a statistic
    def __init__(self, *args, statistic='population', center=(0, 0), **kwargs):
        super().__init__(*args, **kwargs)
        self.statistic = statistic
        self.state.involution = 1  # also do involutions
        self.center = np.array(center, dtype='d') # guarantee a numpy array here
        self.make_involution_lookup_naive()  # instantiate the involution lookup
        # self.make_involution_lookup_lattice(n=40) # TODO rip this back out once debugged
        self.ideal_statistic = sum([node[self.statistic] for node in self.state.graph.nodes().values()])/len(self.state.graph.nodes())
        # self.ideal_statistic = sum([self.state.node_data[node_id][self.statistic]
        #                             for node_id in self.state.node_to_color.keys()]) / len(self.state.color_to_node)
        # assumes balance
        # self.score_log = []


        self._proposal_state = copy.deepcopy(self.state)
        self._proposal_state.involution *= -1 # should be opposite of original state
        self._proposal_state.log_contested_edges = False # don't waste time logging this

    def make_involution_lookup_naive(self):
        self.involution_lookup = dict()
        for edge in self.state.graph.edges:  # these edges are presumed undirected, we have to sort out directionality
            n1 = self.state.graph.nodes()[edge[0]]
            n2 = self.state.graph.nodes()[edge[1]]
            # should this be greater or less than zero?
            if compute_dot_product(n1['Centroid'], n2['Centroid'], center=self.center) >= 0:
                self.involution_lookup[edge] = 1
                self.involution_lookup[(edge[1], edge[0])] = -1
            else:
                self.involution_lookup[edge] = -1
                self.involution_lookup[(edge[1], edge[0])] = 1

    def get_involution(self, state, edge):
        # edge is (node_id, node_id) - need to return in correct order?
        return self.involution_lookup[edge] * state.involution

        # each edge is stored twice, I guess, because we're not certain how it'll be looked up
        # but if its state doesn't match the involution state, we need to flip the order

    # @Decorators.contested_edges_updater # we will figure this out later


    def handle_rejection(self, prop, state):
        super().handle_rejection(prop, state)
        # print("Involution on step {}".format(state.iteration))
        state.involution *= -1
        # state.move_log.append(None) #this is how we're coding an involution at present
        # TODO refactor state_log to allow different types of events

    def get_directed_edges(self, state):
        # does NOT do any checks for connected components, etc
        return [(edge[1], edge[0]) if self.get_involution(state, edge) == -1 else edge for edge in state.contested_edges]


    def proposal(self, state):
        # pick a conflicted edge at random. since this is symmetric under involution, no need to compute Q(z,z')/Q(z',z).

        update_contested_edges(state)

        # TODO do we need to find Q(x,x') here since we're restricting to states that don't disconnect the graph?
        # don't think so
        edges = random.sample(state.contested_edges,
                              len(state.contested_edges))  # TODO this is currently an O(n) operation technically,

        for edge in edges:
            iedge = (edge[1], edge[0]) if self.get_involution(state, edge) == -1 else edge
            old_color, new_color = self.state.node_to_color[iedge[0]], self.state.node_to_color[iedge[1]]  # for clarity
            proposed_smaller = self.state.graph.subgraph(
                [i for i in self.state.color_to_node[old_color] if i != iedge[0]])  # the graph that lost a node
            if not len(proposed_smaller) or not nx.is_connected(
                    proposed_smaller):  # len(proposed_smaller) checks we didn't eliminate the last node in district
                continue  # can't disconnect districts or eliminate districts entirely

            score = self.score_proposal(iedge[0], old_color, new_color, self.state)
            return (iedge[0], new_color), score
        raise RuntimeError("Exceeded tries to find a proposal")


    # def score_proposal(self, node_id, old_color, new_color, state):
    #     # we want to MINIMIZE score
    #     current_score = sum([(i - self.ideal_statistic) ** 2 for i in state.stat_tally[self.statistic].values()])
    #     sum_smaller = sum([state.graph.nodes()[i][self.statistic] for i in state.color_to_node[old_color] if i != node_id])
    #     sum_larger = state.stat_tally[self.statistic][new_color] + state.graph.nodes()[node_id][self.statistic]
    #
    #     new_score = (current_score
    #                  - (state.stat_tally[self.statistic][new_color] - self.ideal_statistic) ** 2
    #                  - state.stat_tally[self.statistic][old_color] ** 2
    #                  + (sum_smaller - self.ideal_statistic) ** 2
    #                  + (sum_larger - self.ideal_statistic) ** 2)
    #
    #     # TODO should beta be owned by state, or by the Markov process?
    #     # technically could be state-dependent when we have parallel tempering
    #
    #     # self.score_log.append(new_score - current_score)  # for debugging purposes
    #
    #     return np.exp(-1 * (new_score - current_score) * self.beta * self.measure_beta)  # TODO double check the sign here


class CenterOfMassFlow(TemperedProposalMixin):
    def __init__(self, *args, weight_attribute=None, center=(0,0), **kwargs):
        super().__init__(*args, **kwargs)
        self.center = np.array(center, dtype='d')
        self.weight_attribute = weight_attribute

        for state in (self.state, self._proposal_state):
            for node_id in state.graph.nodes():
                # TODO parameterize 'Centroid' instead of hardcoding
                # we need to guarantee these are numpy arrays up front
                state.graph.nodes()[node_id]['Centroid'] = np.array(state.graph.nodes()[node_id]['Centroid'])

    def handle_rejection(self, prop, state):
        # TODO need to make this cleaner

        super().handle_rejection(prop, state)
        # print("Involution on step {}".format(state.iteration))
        # state.involution *= -1

    def handle_acceptance(self, prop, state):
        super().handle_acceptance(prop, state)

        # these are all simple objects, so copy is safe - how expensive is it though?
        self.state.contested_nodes = copy.copy(self._proposal_state.contested_nodes)
        self.state.contested_edges = copy.copy(self._proposal_state.contested_edges)
        self.state.com_centroid = copy.copy(self._proposal_state.com_centroid)
        self.state.com_total_weight = copy.copy(self._proposal_state.com_total_weight)
        self.state.articulation_points = copy.copy(self._proposal_state.articulation_points)

        self.state.articulation_points_updated += 1
        self.state.com_updated += 1
        self.state.contested_edges_updated +=1


    def proposal_filter(self, state, proposals):

        update_center_of_mass(state)
        # stack this generator on top of the earlier one
        for proposal in super().proposal_filter(state, proposals):
            # filter out any proposals that don't meet the criterion
            node_id, old_color, new_color = proposal
            (centroid_new_color, centroid_old_color,
             total_weight_new_color, total_weight_old_color) = calculate_com_one_step(
                state, (node_id, old_color, new_color), self.weight_attribute)

            if cython_biconnected:
                # cython version
                dp_old = dot_product(state.com_centroid[old_color], centroid_old_color, self.center)
                dp_new = dot_product(state.com_centroid[new_color], centroid_new_color, self.center)
            else:
                dp_old = compute_dot_product(state.com_centroid[old_color], centroid_old_color, self.center)
                dp_new = compute_dot_product(state.com_centroid[new_color], centroid_new_color, self.center)


            if (dp_old + dp_new)*state.involution > 0:
                yield proposal




class CenterOfMassFlowFixed(CenterOfMassFlow):

    def get_directed_edges(self, state):
        update_contested_edges(state)
        update_center_of_mass(state)

        evaluated_props = set()
        for edge in state.contested_edges:

            # test the default direction - if it isn't valid, return a flipped edge
            node_id = edge[0] # proposal is make edge[0] the same color as edge[1]
            old_color = state.node_to_color[edge[0]]
            new_color = state.node_to_color[edge[1]]
            # proposal is valid if the shift in center of mass is in the direction of the vector field
            if (node_id, new_color) in evaluated_props:
                continue # don't bother with this one, we've already evaluated it


            centroid_new, weights_new = calculate_com_one_step(state, (node_id, old_color, new_color), self.weight_attribute)



            if compute_dot_product(state.com_centroid[new_color], centroid_new[new_color], center=self.center):
                yield edge
            else:
                yield (edge[1], edge[0])
            evaluated_props.add((node_id, new_color))


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


def create_district_boundary_naive(state):

    district_boundary = defaultdict(set)

    for node_id, color in state.node_to_color.items():

        for neighbor in state.graph.neighbors(node_id):
            if state.node_to_color[neighbor]!= color:
                neighbor_color = state.node_to_color[neighbor]
                district_boundary[(min(color, neighbor_color), max(color, neighbor_color))].add(
                    (min(node_id, neighbor), max(node_id, neighbor)))

    return district_boundary



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

                if neighbor_color != new_color:
                    state.district_boundary[key].add((min(neighbor, node_id), max(neighbor, node_id)))
                else: # color == new_color
                    key = (min(neighbor_color, old_color), max(neighbor_color, old_color))
                    state.district_boundary[key].remove((min(neighbor, node_id), max(neighbor, node_id)))

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



class DistrictToDistrictFlow(MetropolisProcess):
    # if not scored_proposals:
    #     return (None, 0, 1), 0  # TODO fix this, it won't generalize

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # assign initial involution state - should this be specifiable by user?
        involution_state = dict()

        for district_id in self.state.color_to_node.keys():
            for other_district_id in self.state.color_to_node.keys():
                if district_id < other_district_id:
                    involution_state[(district_id, other_district_id)] = random.choices([-1, 1])[0]
        self.state.involution_state = involution_state
        self.log_com = False  # TODO repair the issue here


    def involve_state(self, state):
        state.involution_state[min(self.boundary), max(self.boundary)]*=-1

    # def handle_rejection(self, prop, state):
    #     super().handle_rejection(prop, state)
    #     self.involution_state[min(prop[1], prop[2]), max(prop[1], prop[2])]*= -1 # flip the appropriate involution state


    def proposal(self, state):
        try:
            return super().proposal(state)
        except IndexError:
            # occurs when no valid proposal could be identified - we still need to involve correctly
            return (None, self.boundary[0], self.boundary[1]), 0 # no probability of accepting, will involve correctly


    def get_directed_edges(self, state):
        update_contested_edges(state)
        update_district_boundary(state)

        # only allow correctly directed edges from one district boundary we've picked

        # TODO rng here
        counter = 0
        while True:
            boundary = random.choices(list(state.district_boundary.keys()))[0]
            if state.district_boundary[boundary]: # if this isn't an empty list
                break
            counter +=1
            if counter > 100:
                raise ValueError("Didn't select a valid boundary within valid time")


        if self.state.involution_state[boundary] == 1:
            old_color, new_color = boundary

        else:
            # state = -1
            new_color, old_color = boundary

        self.boundary = (old_color, new_color) # always in the direction of flow

        for edge in state.district_boundary[boundary]:
            if state.node_to_color[edge[0]] == old_color:
                yield edge
            else:
                yield (edge[1], edge[0])


class SingleNodeFlip(MetropolisProcess):

    def __init__(self, *args, minimum_population=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.minimum_population = minimum_population

    def proposal(self, state):
        scored_proposals = {}
        # score = self.score_proposal()
        update_contested_edges(state)
        for edge in state.contested_edges:
            for node_id, neighbor in zip(edge, (edge[1], edge[0])): # there are two
                old_color = state.node_to_color[node_id]
                new_color = state.node_to_color[neighbor]

                if (node_id, old_color, new_color) in scored_proposals:
                    continue # already scored this proposal

                if not check_population(state, old_color, minimum=self.minimum_population):
                    continue

                scored_proposals[(node_id, old_color, new_color)] = self.score_proposal(node_id, old_color, new_color, state)

        proposal = random.choices(list(scored_proposals.keys()))[0]
        prob = self.score_to_prob(scored_proposals[proposal]) # should be totally unweighted here
        if not self.check_connected(state, *proposal):
            prob = 0

        return proposal, prob

    def score_proposal(self, node_id, old_color, new_color, state):
        return cut_length_score(state, (node_id, old_color, new_color))



class SingleNodeFlipTempered(SingleNodeFlip):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._proposal_state = copy.deepcopy(self.state) # TODO maybe rip this out into a mixin for


    def handle_rejection(self, prop, state):
        super().handle_rejection(prop, state)
        self._proposal_state.handle_move((prop[0], prop[2], prop[1]))


    def get_proposals(self, state):
        scored_proposals = {}
        # score = self.score_proposal()
        update_contested_edges(state)
        for edge in state.contested_edges:
            for node_id, neighbor in zip(edge, (edge[1], edge[0])): # there are two
                old_color = state.node_to_color[node_id]
                new_color = state.node_to_color[neighbor]

                if (node_id, old_color, new_color) in scored_proposals:
                    continue # already scored this proposal

                if not check_population(state, old_color, minimum=self.minimum_population):
                    continue
                scored_proposals[(node_id, old_color, new_color)] = self.score_proposal(node_id, old_color, new_color, state)
        return scored_proposals


    def proposal(self, state):
        scored_proposals = self.get_proposals(self.state)
        proposal_probs = {k:self.score_to_proposal_prob(v) for k,v in scored_proposals.items()}

        proposal = random.choices(list(proposal_probs.keys()),
                                  weights=proposal_probs.values())[0]

        if not self.check_connected(state, *proposal) and simply_connected(state, *proposal):
            return proposal, 0 # always zero

        self._proposal_state.handle_move(proposal)
        reverse_proposals = self.get_proposals(self._proposal_state)
        reverse_proposal_probs = {k:self.score_to_proposal_prob(v) for k,v in reverse_proposals.items()}

        try:
            q = reverse_proposal_probs[(proposal[0], proposal[2], proposal[1])]/sum(reverse_proposal_probs.values())
            q_prime = proposal_probs[proposal]/sum(proposal_probs.values())
            prob = self.score_to_prob(scored_proposals[proposal])*q/q_prime
            return proposal, prob
        except KeyError:
            return proposal, 0 # this happens sometimes but probably shouldn't for single node flip

class PrecintFlowTempered(PrecintFlow):


    def proposal(self, state):

        # get all the proposals, with their associated probabilities
        proposals = self.get_proposals(state, filter_connected=False)
        # pick a proposal
        try:
            proposal, score = self.pick_proposal(proposals)
        except ValueError:
            # state.involution *=-1
            # self._proposal_state.involution *= -1
            # return self.proposal(state) # if we can't find a valid proposal
            return (None, None, None), 0 # we couldn't find a valid proposal, this is guaranteed to cause involution

        node_id, old_color, new_color = proposal
        self._proposal_state.handle_move(proposal)
        reverse_proposals = self.get_proposals(self._proposal_state, filter_connected=True)
        reverse_proposal_probs = {k:self.score_to_proposal_prob(v) for k,v in reverse_proposals.items()}
        proposal_probs = {k: self.score_to_proposal_prob(v) for k,v in proposals.items()}

        try:
            q_prime = reverse_proposals[(node_id, new_color, old_color)]/sum(reverse_proposal_probs.values())
            q = proposals[proposal]/sum(proposal_probs.values())
            return proposal, q_prime/q*exp(-score*self.beta*self.measure_beta) # TODO should this be a different func?
        except KeyError:
            return proposal, 0 # sometimes reverse is not contained in proposals list


    def handle_rejection(self, prop, state):
        super().handle_rejection(prop, state)
        if prop[0] is not None: # we couldn't find a valid proposal
            self._proposal_state.handle_move((prop[0], prop[2], prop[1])) # flip back to old color
        self._proposal_state.involution *= -1

    def pick_proposal(self, proposals):

        # threshold = np.random.random()
        # keys = sorted(proposals.keys())
        counter = 0
        while True:
            proposal = random.choices(list(proposals.keys()), weights=self.score_to_proposal_prob(proposals.values()), k=1)[0]
            if self.check_connected(self.state, proposal[0], proposal[1], proposal[2]):
                return proposal, proposals[proposal]
            else:
                counter +=1
                proposals[proposal] = 0 # this wasn't connected, so we can't pick it
                if counter >= len(proposals):
                    # return (None, None, None), 0 # not actually a valid proposal, guaranteed to result in involution
                    raise ValueError("Could not find connected proposal")

                proposals[proposal] = 0 # probability is zero if it's not connected

    def get_proposals(self, state=None, filter_connected=False):

        update_contested_edges(state)
        update_perimeter_aggressive(state)
        proposals = defaultdict(int)

        directed_edges = self.get_directed_edges(state)
        for iedge in directed_edges:

            old_color = state.node_to_color[iedge[0]]  # find the district that the original belongs to
            new_color = state.node_to_color[iedge[1]]

            if not check_population(state, old_color, minimum=500):
                continue # ensure that population of either does not drop below 500
                # TODO parameterize this

            if (iedge[0], old_color, new_color) in proposals:
                continue # we've already scored this proposal, no need to redo

            if filter_connected and not self.check_connected(state, iedge[0], old_color, new_color):
                 continue

            score = self.score_proposal(iedge[0], old_color, new_color, state)  # smaller score is better
            proposals[(iedge[0], old_color, new_color)] = score

        return proposals

class PrecinctFlowNoConnectedTemper(PrecintFlowTempered):
    # will not temper proposals based on connected-ness - connectedness only checked at accept/reject time

    def get_proposals(self, state=None):
        # no option to filter connected here
        return super().get_proposals(state, filter_connected=False)


    def proposal(self, state):
        # TODO this is common with SingleNodeFlipTempered, perhaps split this out
        scored_proposals = self.get_proposals(self.state)
        proposal_probs = {k:self.score_to_proposal_prob(v) for k,v in scored_proposals.items()}

        proposal = random.choices(list(proposal_probs.keys()),
                                  weights=proposal_probs.values())[0]

        self._proposal_state.handle_move(proposal)
        if not self.check_connected(state, *proposal) and simply_connected(state, *proposal):
            return proposal, 0 # always zero

        reverse_proposals = self.get_proposals(self._proposal_state)
        reverse_proposal_probs = {k:self.score_to_proposal_prob(v) for k,v in reverse_proposals.items()}

        try:
            # TODO this should match q, q_prime nomenclature in superclass - still correct, need to swap names
            q = reverse_proposal_probs[(proposal[0], proposal[2], proposal[1])]/sum(reverse_proposal_probs.values())
            q_prime = proposal_probs[proposal]/sum(proposal_probs.values())
            prob = self.score_to_prob(scored_proposals[proposal])*q/q_prime
            return proposal, prob
        except KeyError:
            return proposal, 0 # this happens sometimes but probably shouldn't for single node flip


class CenterOfMassLazyInvolution(CenterOfMassFlow):
    def __init__(self, *args, involution_rate=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.involution_rate = involution_rate

    def perform_involution(self):
        # TODO rng here
        if np.random.uniform(size=1) < self.involution_rate:
            super().perform_involution() # involve


class DistrictToDistrictLazyInvolution(DistrictToDistrictFlow):
    def __init__(self, *args, involution_rate=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.involution_rate = involution_rate

    def perform_involution(self):
        # TODO rng here
        if np.random.uniform(size=1) < self.involution_rate:
            super().perform_involution() # involve



# TODO this class is broken
class LazyAdjustedInvolutionMixin(object):
    # automatically adjusts the involution rate along a sigmoid function
    # every time we reject a proposal, we will decrease the involution rate - if we're making bad proposals
    # we need to move to another state
    # similarly, every time we accept a proposal, we should slow down, and search the surrounding area


    def __init__(self, *args, involution_rate_delta=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.involution_rate_delta = involution_rate_delta

    # use logit functions
    @staticmethod
    def sigmoid_to_absolute(x):
        return np.log(x/(1-x))

    @staticmethod
    def absolute_to_sigmoid(z):
        return 1/(1+np.exp(-z))

    def handle_rejection(self, prop, state):
        super().handle_rejection(prop, state)
        z = LazyAdjustedInvolutionMixin.sigmoid_to_absolute(self.involution_rate)
        z -= self.involution_rate_delta
        self.involution_rate = LazyAdjustedInvolutionMixin.absolute_to_sigmoid(z)


    def handle_acceptance(self, prop, state):
        super().handle_acceptance(prop, state)
        z = LazyAdjustedInvolutionMixin.sigmoid_to_absolute(self.involution_rate)
        z += self.involution_rate_delta
        self.involution_rate = LazyAdjustedInvolutionMixin.absolute_to_sigmoid(z)


# assumes there is a weight, x, y attribute associated with each node
def compute_com(state, district_id):
    # TODO needs updating for new handling of node data
    # get an updated center of mass for a particular district_id

    nodes = state.color_to_node[district_id]  # set
    com_x = sum([state.node_data[node_id]['weight'] * state.node_data[node_id]['x'] for node_id in nodes])
    com_y = sum([state.node_data[node_id]['weight'] * state.node_data[node_id]['y'] for node_id in nodes])

    sum_weights = sum([state.node_id[node_id]['weight'] for node_id in nodes])
    # nodes = [i for i in ]

    return com_x / sum_weights, com_y / sum_weights


def com_valid_moves(state, district_id, center=(0, 0)):
    # determine list of neighboring (contested) nodes that could be colored the same as the given district
    # returns list[node_id]
    nodes = state.color_to_node[district_id]
    cont_nodes = {i[1] for i in state.contested_edges if i[0] in nodes}  # list of node_ids neighboring
    # com_original =

    for edge in cont_nodes:
        yield edge



# TODO we can numba this function if needed
def compute_dot_product(a, b, center=(0, 0), normalize=True):
    # this function's name is a bit of a misnomer
    # a, b, center are (x,y) points
    # find vector from a to b, compute dot product against vector perpendicular to the midpoint
    a, b, center = np.matrix(a).T, np.matrix(b).T, np.matrix(center).T  # guarantee an array
    vec_a_b = b - a
    midpoint = a + vec_a_b / 2
    vec_from_center = midpoint - center
    vec_perp_center = np.matmul(ROT_MATRIX,
                                vec_from_center)  # perform a 90-degree CCW rotation to find flow at midpoint
    dp = np.dot(vec_a_b.T, vec_perp_center)  # find the dot product

    if normalize:
        return dp/np.linalg.norm(vec_perp_center)/np.linalg.norm(vec_a_b)
    else:
        return  dp  # dot product


def center_of_mass_updater(func):
    # ensure that state has an updated copy of center of mass before this function runs
    def inner(state, *args, **kwargs):
        if not hasattr(state, 'district_to_com'):
            state.district_to_com = dict()  # TODO fix this to init correctly
            state.district_to_com_updated = state.iteration

        if state.iteration != state.district_to_com_updated:

            # TODO this isn't correct - also need to update the old_color districts
            districts_to_update = {new_color for node_id, old_color, new_color in
                                   state.state_log[state.district_to_com_updated:]}
            for district_id in districts_to_update:
                state.district_to_com[district_id] = compute_com(state, district_id)
        return func(state, *args, **kwargs)

    return inner


def perimeter_naive(state):
    # TODO refactor
    dd = collections.defaultdict(int)

    for n0, n1 in state.contested_edges:
        shared_length = state.graph.edges()[(n0,n1)]['BorderLength']
        dd[state.node_to_color[n0]] += shared_length
        dd[state.node_to_color[n1]] += shared_length

    return dd


def update_perimeter_aggressive(state):

    # this version assumes that this will get run EVERY time a node is flipped
    update_contested_edges(state) # guarantee contested edges updated before proceeding

    if not hasattr(state, 'district_to_perimeter'):
        state.district_to_perimeter = perimeter_naive(state)
        state.perimeter_updated = state.iteration  # set to current iteration

    for move in state.move_log[state.perimeter_updated:]:

        if move is None:
            continue

        node_id, old_color, new_color = move

        for neighbor in state.graph.neighbors(node_id):
            if neighbor in state.color_to_node[new_color]:
                # we need to reduce the perimeter of new_color by their shared amount
                state.district_to_perimeter[new_color] -= state.graph.edges[(node_id, neighbor)]['BorderLength']

            elif neighbor in state.color_to_node[old_color]:
                # we need to increase the perimeter of old_color by their shared amount
                state.district_to_perimeter[old_color] += state.graph.edges[(node_id, neighbor)]['BorderLength']

            else:
                # we need to increase the perimeter of new_color AND decrease of old color. no change to the perimeter of the 3rd district
                state.district_to_perimeter[new_color] += state.graph.edges[(node_id, neighbor)]['BorderLength']
                state.district_to_perimeter[old_color] -= state.graph.edges[(node_id, neighbor)]['BorderLength']

    state.perimeter_updated = state.iteration


def contested_nodes_naive(state):
    contested = set()
    for node_id in state.graph.nodes():
        color = state.node_to_color[node_id]
        if not all(color==state.node_to_color[other_id] for other_id in nx.neighbors(state.graph, node_id)):
            contested.add(node_id)
    return contested


def log_contested_edges(state):
    # maybe manage an array so we don't need to use these horrible loops
    for edge in state.contested_edges:
        # contested_nodes = set(itertools.chain(*state.contested_edges)) # this is horrible, maybe start tracking contested nodes?
        state.contested_edge_counter[edge] += 1

    for node in state.contested_nodes:
        state.contested_node_counter[node] += 1


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


def compactness_score(state, proposal):
    prop_node, old_color, new_color = proposal # unpack
    score, score_prop = 0, 0

    area_prop = state.graph.nodes()[prop_node]['Area'] if 'Area' in state.graph.nodes()[prop_node] else 1

    perim_smaller = state.district_to_perimeter[old_color]
    area_smaller = sum([(state.graph.nodes()[node_id]['Area'] if 'Area' in state.graph.nodes()[node_id] else 1)
                     for node_id in state.color_to_node[old_color]]) # if we don't have an Area, just weight evenly

    perim_larger = state.district_to_perimeter[new_color]
    area_larger = sum([(state.graph.nodes()[node_id]['Area'] if 'Area' in state.graph.nodes()[node_id] else 1)
                     for node_id in state.color_to_node[new_color]]) # if we don't have an Area, just weight evenly

    node_neighbors = state.graph.neighbors(prop_node)

    perim_larger_new = perim_larger + sum([state.graph.edges()[(prop_node, other_node)]['BorderLength'] for other_node in node_neighbors
                                           if other_node not in state.color_to_node[new_color]])
    perim_smaller_new = perim_smaller - sum([state.graph.edges()[(prop_node, other_node)]['BorderLength'] for other_node in node_neighbors
                                           if other_node not in state.color_to_node[old_color]])

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

def connected_breadth_first(state, node_id, old_color):

    district_nodes = state.color_to_node[old_color].difference({node_id}) # all of the old_color nodes without the one we're removing
    to_connect = district_nodes.intersection(set(state.graph.neighbors(node_id))) # need to show each of these are still connected

    if not to_connect: # TODO should this check get run here or should it be guaranteed?
        return False # we deleted the last node of a district, this is illegal

    init = to_connect.pop()
    to_search = [init] # ordered list of nodes available to search
    visited_set = {init}

    connected_counter = 0

    while True: # there are still nodes we haven't found
        connected_counter += 1

        # evaluate halting criteria
        if not to_connect - visited_set: # we visited all of the nodes we needed to search
            state.connected_true_counter.append(connected_counter)
            return True
        if not to_search: # no more nodes available to search
            # raise ValueError("")
            state.connected_false_counter.append(connected_counter)
            return False

        # new nodes that we haven't seen yet
        new_neighbors = set(state.graph.neighbors(to_search.pop())).intersection(district_nodes).difference(visited_set)

        priority_neighbors = to_connect.intersection(new_neighbors)
        nonpriority_neighbors = new_neighbors - priority_neighbors
        visited_set.update(new_neighbors) # seen new neighbors
        to_search.extend(nonpriority_neighbors)
        to_search.extend(priority_neighbors) # ensure that priority neighbors are at the top of the queue


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

def update_boundary_nodes_naive(state):
    counter = collections.Counter([v for k,v in state.node_to_color.items() if state.graph.nodes()[v]['boundary']])
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


def count_node_flips(process):

    # count the number of times each node was flipped
    return collections.Counter([i[0] for i in process.state.move_log if i is not None])



def simply_connected(state, node_id, old_color, new_color):
    # asssuming that a proposal will result in a graph being connected, check that it will be simply connected
    # this occurs iff there is another node district:old_color that touches either boundary or another color
    # update_contested_edges(state) # do we need this here or can we pull it somewhere else?

    if state.boundary_node_counter[old_color] > 0:
        return True # this handles the vast majority of cases

    smaller = state.color_to_node[old_color] - {node_id}
    contested_nodes = {i[0] for i in state.contested_edges}.union({i[1] for i in state.contested_edges}).intersection(
        smaller) # nodes that are contested and are in the shrunk district
    for node in contested_nodes:
        # is this node either on the boundary, or does it touch some other color than
        if state.graph.nodes()[node]['boundary'] or len([i for i in state.graph.neighbors(node)
                                                         if state.node_to_color[i] not in (old_color, new_color)]) > 0:

            # about to return True - should be able to check that

            return True
    return False


def check_population(state, old_color, minimum=400):
    return (len(state.color_to_node[old_color]) - 1) > minimum
