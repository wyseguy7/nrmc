import logging
import networkx as nx
from collections import namedtuple, defaultdict
import numpy as np
import copy
import random
import collections

# Remaining to-dos
# Sort out state initializer, object loader
# Details on COM flow algorithm

####### Meeting notes 2019-12-06:
# Make 'state' with listeners that track need for metadata

# state object that allows arbitrary extensions
ROT_MATRIX = np.matrix([[0, -1], [1, 0]])
exp = lambda x: np.exp(min(x, 700)) # avoid overflow

class Decorators(object):

    @classmethod
    def contested_edges_updater(cls, func):

        # before performing function, checks that 'state' contains an updated copy, and updates as needed.

        def inner(state, *args, **kwargs):
            # init to current state
            if not hasattr(state, 'contested_edges'):
                state.contested_edges = contested_edges_naive(state)
                state.contested_edges_updated = state.iteration  # set to current iteration

            # this may be an empty list if it's already been updated
            for node_id, old_color, new_color in state.state_log[state.contested_edges_updated:]:
                # move is provided as (node_id, color_id)
                neighbors = state.graph.edges(node_id)
                # edges to add
                state.contested_edges.update(
                    {(min(u, v), max(u, v)) for u, v in neighbors if state.node_to_color[v] != new_color})
                # edges to remove
                state.contested_edges.difference_update(
                    {(min(u, v), max(u, v)) for u, v in neighbors if state.node_to_color[v] == new_color})

                #     # at some point it will be more efficient to just naively reconstruct the contested edges, we should look out for this
                state.contested_edges_updated = state.iteration
            # note that func must accept state as the FIRST argument. will this impact our ability to chain these together?
            return func(state, *args, **kwargs)

        return inner


class State(object):

    def __init__(self, graph, coloring, tallied_stats=('node_count','population')):

        self.graph = graph

        self.node_to_color = coloring
        d = defaultdict(set)
        for node_id, district_id in coloring.items():
            d[district_id].add(node_id)
        self.color_to_node = dict(d)

        self.iteration = 0
        self.state_log = []

        self.stat_tally = defaultdict(dict)

        for district_id, nodes in self.color_to_node.items():
            for stat in tallied_stats:
                self.stat_tally[stat][district_id] = sum(graph[node_id][stat] for node_id in nodes)

        self.tallied_stats = tallied_stats


    # def __init__old(self, edges, coloring, node_data=None, edge_data=None, tallied_stats=('node_count', 'population')):
    #     # nodes are node ids, edges iterable of (node_id, node_id), coloring is (node_id, district_id)
    #     # node_id, district_id must be orderable, hashable
    #     # node_data, edge_data are optional, {node_id: dict} lookups
    #     # tallied stats will be updated after each iteration
    #     self.__dict__ = {}
    #
    #     self.node_data = node_data  # this is unchecked at present
    #     self.edge_data = edge_data  # make sure this is easily accessible
    #
    #     nodes = {edge[0] for edge in edges}.union({edge[1] for edge in edges})
    #     if nodes.difference(set(coloring.keys())) or set(coloring.keys()).difference(nodes):
    #         raise ValueError("Edges and coloring must match")
    #
    #     self.graph = nx.Graph()
    #     self.graph.add_edges_from(edges)  # hopefully this will occur lazily
    #     self.node_to_color = coloring
    #     # print(self.graph)
    #     d = defaultdict(set)
    #     for node_id, district_id in coloring.items():
    #         d[district_id].add(node_id)
    #     self.color_to_node = dict(d)
    #
    #     # frequently we need to keep track of summed fields, we have a special way of doing that
    #     self.stat_tally = defaultdict(dict)
    #     for district_id, nodes in self.color_to_node.items():
    #         for stat in tallied_stats:
    #             self.stat_tally[stat][district_id] = sum(self.node_data[node_id][stat] for node_id in nodes)
    #
    #     self.tallied_stats = tallied_stats
    #
    #     self.state_log = []  # contains (node_id, old_color, new_color) pairs of moves, in order
    #     self.iteration = 0  # this will get bumped each time we make a move

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def flip(self, node_id, district_id):

        # update lookups
        old_color = self.node_to_color[node_id]
        self.color_to_node[old_color].remove(node_id)
        self.color_to_node[district_id].add(node_id)
        self.node_to_color[node_id] = district_id

        # record the move
        self.state_log.append((node_id, old_color, district_id))

        self.iteration += 1  # add to iteration

        for stat in self.tallied_stats:
            # the stat tally giveth
            self.stat_tally[stat][district_id] += self.node_data[node_id][stat]
            # ...and it taketh away
            self.stat_tally[stat][old_color] -= self.node_data[node_id][stat]

    # shim to add in data

    @classmethod
    def from_file_hardcode(cls):
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



    # @classmethod
    # def from_square_lattice(cls, n_x=40, n_y=40):
    #     ''' n_x, n_y: specify the 'length' and 'width' of the graph, respectively. i.e. n_x=40, n_y=30 specifies a 40x30 square lattice'''
    #
    #     edges = []
    #     nodes = []
    #     # x_pos = []
    #     # y_pos = []
    #     node_data = {}
    #     edge_data = {}
    #
    #     for ix in range(n_x + 1):
    #         for iy in range(n_y + 1):
    #             ni = ix + (n_x + 1) * iy
    #             nodes.append(ni)
    #             node_data[ni] = {"population": 1, "x": ix, "y": iy}
    #
    #     for ix in range(n_x):
    #         for iy in range(n_y):
    #             ni = ix + (n_x + 1) * iy
    #             nipx = (ix + 1) + (n_x + 1) * iy
    #             nipy = ix + (n_x + 1) * (iy + 1)
    #
    #             edges.append((ni, nipx))
    #             edges.append((ni, nipy))
    #             edge_data[(ni, nipy)] = {"border_length": 1}
    #             edge_data[(ni, nipx)] = {"border_length": 1}
    #
    #     for ix in range(n_x):
    #         ni = ix + (n_x + 1) * n_y
    #         nipx = ix + 1 + (n_x + 1) * n_y
    #
    #         edges.append((ni, nipx))
    #         edge_data[(ni, nipx)] = {"border_length": 1}
    #
    #     for iy in range(n_y):
    #         ni = n_x + (n_x + 1) * iy
    #         nipy = n_x + (n_x + 1) * (iy + 1)
    #         edges.append((ni,nipy))
    #         edge_data[(ni, nipy)] = {"border_length": 1}
    #
    # # initial state divides "left" and "right" - should we do this stochastically?
    #     coloring = {i:(1 + int(i>len(nodes))) for i in nodes}
    #     node_data = {i: {"population": 1} for i in nodes}
    #     edge_data = {i: {"border_length": 1} for i in edges}
    #
    #     return State(edges, coloring, node_data, edge_data, tallied_stats=('population'))

       #  def __init__(self, edges, coloring, node_data=None, edge_data=None, tallied_stats=('node_count', 'population')):

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


# we can either subclass this object or attach a bunch of functions to it, either is fine really
class MetropolisProcess(object):

    def __init__(self, state):
        self._initial_state = copy.deepcopy(state)  # save for later
        self.state = state

    def proposal(self):
        pass

    def handle_acceptance(self, prop, state):
        state.flip(prop[0], prop[1])  # we can still customize this but this ought to be sufficient for a lot of cases

    def handle_rejection(self, prop, state):
        pass  # this is algorithm-specific

    def accept_reject(self, score):
        # classic Metropolis accept-reject. we can override if needed
        # TODO rng here
        # returns a boolean
        u = np.random.uniform()
        return u < min(score, 1)

    def step(self):
        prop, score = self.proposal()  # no side effects here, should be totally based on current state

        if self.accept_reject(score):  # no side effects here
            # proposal accepted!
            self.handle_acceptance(prop, self.state)  # now side effects
        else:
            self.handle_rejection(prop, self.state)  # side effects also here

    def check_connected(self, state, node_id, old_color):

        # TODO change this to a set
        proposed_smaller = state.graph.subgraph(
            [i for i in state.color_to_node[old_color] if i != node_id])  # the graph that lost a node
        return len(proposed_smaller) and nx.is_connected(proposed_smaller)


class PrecintFlow(MetropolisProcess):
    # TODO perhaps stat_tally should go in here instead

    # make proposals on what maximizes the L2 norm of a statistic
    def __init__(self, state, statistic='population', center=(0, 0), lmda=1e-8):
        super().__init__(state)
        self.statistic = statistic
        self.state.involution = 1  # also do involutions
        self.center = center
        # print("ping")
        self.make_involution_lookup_naive()  # instantiate the involution lookup
        self.ideal_statistic = sum([self.state.node_data[node_id][self.statistic]
                                    for node_id in self.state.node_to_color.keys()]) / len(self.state.color_to_node)
        # assumes balance
        self.lmda = lmda  # used to normalize the - rigorous way to set this? rule of 0.23?
        self.score_log = []

    def score_to_prob(self, score):
        return exp(-0.5*self.lmda*score)

    def make_involution_lookup_naive(self):
        self.involution_lookup = dict()
        for edge in self.state.graph.edges:  # these edges are presumed undirected, we have to sort out directionality
            # print("pong")
            n1 = self.state.node_data[edge[0]]  # the centerpoint between the two nodes
            n2 = self.state.node_data[edge[1]]

            # centroid = ((n1.x+n2.x)/2, (n1.y+n2.y)/2)

            # theta1 = np.math.atan2(n1.y - centroid[1], n1.x - centroid[0])
            # theta2 = np.math.atan2(n2.y - centroid[1], n2.x - centroid[0])

            # we will always add two entries to the involution lookup per edge

            # should this be greater or less than zero?
            if compute_dot_product((n1['x'], n1['y']), (n2['x'], n2['y']), center=self.center) >= 0:
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
        state.involution *= -1
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


    def score_proposal(self, node_id, old_color, new_color, state):
        # we want to MINIMIZE score
        current_score = sum([(i - self.ideal_statistic) ** 2 for i in state.stat_tally[self.statistic].values()])
        sum_smaller = sum([state.node_data[i][self.statistic] for i in state.color_to_node[old_color] if i != node_id])
        sum_larger = state.stat_tally[self.statistic][new_color] + state.node_data[node_id][self.statistic]

        new_score = (current_score
                     - (state.stat_tally[self.statistic][new_color] - self.ideal_statistic) ** 2
                     - state.stat_tally[self.statistic][old_color] ** 2
                     + (sum_smaller - self.ideal_statistic) ** 2
                     + (sum_larger - self.ideal_statistic) ** 2)

        # TODO should beta be owned by state, or by the Markov process?
        # technically could be state-dependent when we have parallel tempering

        self.score_log.append(new_score - current_score)  # for debugging purposes

        return np.exp(-1 * (new_score - current_score) * self.lmbda)  # TODO double check the sign here

    def score_proposal_old(self, node_id, old_color, new_color, state):
        # scores based on summed L2 norm of population (so an even spread will minimize)
        # equivalent to comparing

        current_score = sum([i ** 2 for i in state.stat_tally[self.statistic].values()])
        sum_smaller = sum([state.node_data[i][self.statistic] for i in state.color_to_node[old_color] if i != node_id])
        sum_larger = state.stat_tally[self.statistic][new_color] + state.node_data[node_id][self.statistic]
        new_score = current_score - state.stat_tally[self.statistic][new_color] ** 2 - state.stat_tally[self.statistic][
            old_color] ** 2 + sum_smaller ** 2 + sum_larger ** 2
        # print(np.sqrt(new_score)-np.sqrt(current_score))
        return np.exp((np.sqrt(new_score) - np.sqrt(current_score)) / 5000)
#
#         # TODO this feels clumsy, mostly just for demonstration purposes
#
#     # scores based on
#
#
class PrecintFlowTempered(PrecintFlow):


    def reverse_proposal_prob(self, state, proposal):
        node_id, old_color, new_color = proposal
        new_state = copy.deepcopy(state)
        new_state.involution *= -1 # involve
        new_state.flip(node_id, new_color)

        return self.get_proposals(new_state)


        # finds probability of proposing x' -> x

    def score_proposal(self, node_id, old_color, new_color, state):
        update_perimeter_aggressive(state)
        return compactness_score((node_id, old_color, new_color))


    def proposal(self, state):

        # get all the proposals, with their associated probabilities
        proposals = self.get_proposals(state)
        # pick a proposal
        proposal, q = self.pick_proposal(proposals)
        reverse_proposals = self.reverse_proposal_prob(state, proposal)
        q_prime = reverse_proposals[proposal]

        score = self.score_proposal(proposal[0], proposal[1], proposal[2], state)


        return proposal, q/q_prime*exp(-score*self.lmda)



    def pick_proposal(self, proposals):

        threshold = np.random.random()
        cum_sum = 0
        for node_id, prob in proposals:
            cum_sum += prob
            if cum_sum >= threshold:
                return node_id, prob


    def get_proposals(self, state=None):

        update_contested_edges(state)

        proposals = defaultdict(int)
        for iedge in self.get_directed_edges(state):

                old_color = state.node_to_color[iedge[0]]  # find the district that the original belongs to
                new_color = state.node_to_color[iedge[1]]
                self.check_connected(state, iedge[0], old_color)

                # TODO population constraint enforcement - optional, min/max check
                # TODO constraint on compactness -
                # TODO constraint on traversals - figure out later

                score = self.score_proposal(iedge[0], old_color, state.node_to_color[iedge[1]],
                                            state)  # smaller score is better
                proposals[(iedge[0], old_color, new_color)] += score

        # pick a proposal - this assures ordering
        prop_keys_list = list(proposals.keys())
        probs = [self.score_to_prob(proposals[i]) for i in prop_keys_list] #
        prob_sum = sum(probs)

        # idx = np.random.choice(range(len(prop_keys_list)), p=[i /prob_sum for i in probs])
        # choice = prop_keys_list[idx]

        # return choice, probs[idx]  # proposal, probability
        # return zip(prop_keys_list, [prob/prob_sum for prob in probs])
        return {prop:prob/prob_sum for prop, prob in zip(prop_keys_list, probs)}



# assumes there is a weight, x, y attribute associated with each node
def compute_com(state, district_id):
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
def compute_dot_product(a, b, center=(0, 0)):
    # a, b, center are (x,y) points
    # find vector from a to b, compute dot product
    a, b, center = np.matrix(a).T, np.matrix(b).T, np.matrix(center).T  # guarantee an array
    vec_a_b = b - a
    midpoint = a + vec_a_b / 2
    vec_from_center = midpoint - center
    vec_perp_center = np.matmul(ROT_MATRIX,
                                vec_from_center)  # perform a 90-degree CCW rotation to find flow at midpoint
    dp = np.dot(vec_a_b.T, vec_perp_center)  # find the dot product
    return dp  # dot product


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
        shared_length = state.node_data[n0]['shared_perimeter'][n1]  #
        dd[state.node_to_color[n0]] += shared_length
        dd[state.node_to_color[n1]] += shared_length

    return dd

    # requires that contested edges are updated



def update_perimeter_aggressive(state):

    # this version assumes that this will get run EVERY time a node is flipped
    update_contested_edges(state) # guarantee contested edges updated before proceeding

    if not hasattr(state, 'district_to_perimeter'):
        state.district_to_perimeter = perimeter_naive(state)
        state.perimeter_updated = state.iteration  # set to current iteration

    for node_id, old_color, new_color in state.state_log[state.perimeter_updated:]:

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

def update_contested_edges(state):
    if not hasattr(state, 'contested_edges'):
        state.contested_edges = contested_edges_naive(state)
        state.contested_edges_updated = state.iteration  # set to current iteration

    # this may be an empty list if it's already been updated
    for node_id, old_color, new_color in state.state_log[state.contested_edges_updated:]:
        # move is provided as (node_id, color_id)
        neighbors = state.graph.edges(node_id)
        # edges to add
        state.contested_edges.update(
            {(min(u, v), max(u, v)) for u, v in neighbors if state.node_to_color[v] != new_color})
        # edges to remove
        state.contested_edges.difference_update(
            {(min(u, v), max(u, v)) for u, v in neighbors if state.node_to_color[v] == new_color})

    #     # at some point it will be more efficient to just naively reconstruct the contested edges, we should look out for this
    state.contested_edges_updated = state.iteration


def compactness_score(state, proposal):
    prop_node, old_color, new_color = proposal # unpack
    score, score_prop = 0, 0

    area_prop = state.graph.nodes()[prop_node]['Area'] if 'Area' in state.graph.nodes()[prop_node]['Area'] else 1

    perim_smaller = state.district_to_perimeter[old_color]
    area_smaller = sum([(state.graph.nodes()[node_id]['Area'] if 'Area' in state.graph.nodes()[node_id] else 1)
                     for node_id in state.color_to_node[old_color]]) # if we don't have an Area, just weight evenly

    perim_larger = state.district_to_perimeter[new_color]
    area_larger = sum([(state.graph.nodes()[node_id]['Area'] if 'Area' in state.graph.nodes()[node_id] else 1)
                     for node_id in state.color_to_node[new_color]]) # if we don't have an Area, just weight evenly

    node_neighbors = state.graph.neighbors(prop_node)

    perim_larger_new = perim_larger + sum([state.graph.edges()[(prop_node, other_node)] for other_node in node_neighbors
                                           if other_node not in state.color_to_node[new_color]])
    perim_smaller_new = perim_smaller - sum([state.graph.edges()[(prop_node, other_node)] for other_node in node_neighbors
                                           if other_node not in state.color_to_node[old_color]])

    score_old = perim_smaller**2/area_smaller + perim_larger**2/area_larger
    score_new = perim_smaller_new**2/(area_smaller-area_prop) + perim_larger_new**2/(area_larger+area_prop)

    return score_new-score_old # the delta
