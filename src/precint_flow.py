import logging
import networkx as nx
from collections import namedtuple, defaultdict
import numpy as np
import copy

# Remaining to-dos
# Sort out state initializer, object loader
# Details on COM flow algorithm

####### Meeting notes 2019-12-06:
# Make 'state' with listeners that track need for metadata

# state object that allows arbitrary extensions
ROT_MATRIX = np.matrix([[0, -1], [1, 0]])


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
            for node_id, district_id in state.state_log[state.contested_edges_updated:]:
                # move is provided as (node_id, color_id)
                neighbors = state.graph.edges(node_id)
                # edges to add
                state.contested_edges.update(
                    {(min(u, v), max(u, v)) for u, v in neighbors if state.node_to_color[v] != district_id})
                # edges to remove
                state.contested_edges.difference_update(
                    {(min(u, v), max(u, v)) for u, v in neighbors if state.node_to_color[v] == district_id})

                #     # at some point it will be more efficient to just naively reconstruct the contested edges, we should look out for this
                state.contested_edges_updated = state.iteration
            # note that func must accept state as the FIRST argument. will this impact our ability to chain these together?
            return func(state, *args, **kwargs)

        return inner


class State(object):

    def __init__(self, edges, coloring, node_data=None, edge_data=None, tallied_stats=('node_count', 'population')):
        # nodes are node ids, edges iterable of (node_id, node_id), coloring is (node_id, district_id)
        # node_id, district_id must be orderable, hashable
        # node_data, edge_data are optional, {node_id: dict} lookups
        # tallied stats will be updated after each iteration
        self.__dict__ = {}

        self.node_data = node_data  # this is unchecked at present
        self.edge_data = edge_data  # make sure this is easily accessible

        nodes = {edge[0] for edge in edges}.union({edge[1] for edge in edges})
        if nodes.difference(set(coloring.keys())) or set(coloring.keys()).difference(nodes):
            raise ValueError("Edges and coloring must match")

        self.graph = nx.Graph()
        self.graph.add_edges_from(edges)  # hopefully this will occur lazily
        self.node_to_color = coloring
        print(self.graph)
        d = defaultdict(set)
        for node_id, district_id in coloring.items():
            d[district_id].add(node_id)
        self.color_to_node = dict(d)

        # frequently we need to keep track of summed fields, we have a special way of doing that
        self.stat_tally = defaultdict(dict)
        for district_id, nodes in self.color_to_node.items():
            for stat in tallied_stats:
                self.stat_tally[stat][district_id] = sum(self.node_data[node_id][stat] for node_id in nodes)

        self.tallied_stats = tallied_stats

        self.state_log = []  # contains (node_id, district_id) pairs of moves, in order
        self.iteration = 0  # this will get bumped each time we make a move

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def flip(self, node_id, district_id):

        # update lookups
        old_color = self.node_to_color[node_id]
        self.color_to_node[old_color].remove(node_id)
        self.color_to_node[district_id].add(node_id)
        self.node_to_color[node_id] = district_id

        # record the move
        self.state_log.append((node_id, district_id))

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

        return State(edges, coloring=coloring, node_data=node_data)


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
        self._initial_state = copy.deepcopy(state)  # save for later?
        self.state = state

    def proposal(self, state):
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
        prop, score = self.proposal(self.state)  # no side effects here, should be totally based on current state

        if self.accept_reject(score):  # no side effects here
            # proposal accepted!
            self.handle_acceptance(prop, self.state)  # now side effects
        else:
            self.handle_rejection(prop, self.state)  # side effects also here


class PrecintFlowPopulationBalance(MetropolisProcess):
    # TODO perhaps stat_tally should go in stead instead

    # make proposals on what maximizes the L2 norm of a statistic
    def __init__(self, state, statistic='population', center=(0, 0)):
        super().__init__(state)  # TODO am i doing this right in py36?
        self.statistic = statistic
        self.state.involution = 1  # also do involutions
        self.center = center
        # print("ping")
        self.make_involution_lookup_naive()  # instantiate the involution lookup

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

    def get_involution(self, edge):
        # edge is (node_id, node_id) - need to return in correct order?
        return self.involution_lookup[
                   edge] * self.state.involution  # each edge is stored twice, I guess, because we're not certain how it'll be looked up
        # but if its state doesn't match the involution state, we need to flip the order

    # @Decorators.contested_edges_updater # we will figure this out later
    def proposal(self, state=None):
        if not hasattr(state, 'contested_edges'):
            state.contested_edges = contested_edges_naive(state)
            state.contested_edges_updated = state.iteration  # set to current iteration

        # this may be an empty list if it's already been updated
        for node_id, district_id in state.state_log[state.contested_edges_updated:]:
            # move is provided as (node_id, color_id)
            neighbors = state.graph.edges(node_id)
            # edges to add
            state.contested_edges.update(
                {(min(u, v), max(u, v)) for u, v in neighbors if state.node_to_color[v] != district_id})
            # edges to remove
            state.contested_edges.difference_update(
                {(min(u, v), max(u, v)) for u, v in neighbors if state.node_to_color[v] == district_id})

            #     # at some point it will be more efficient to just naively reconstruct the contested edges, we should look out for this
            state.contested_edges_updated = state.iteration

        proposals = defaultdict(int)
        for edge in state.graph.edges:
            if (
            min(edge), max(edge)) in state.contested_edges:  # do we have to do min/max, or will it keep it consistent?

                # make a copy of edge with the correct ordering
                iedge = (edge[1], edge[0]) if self.get_involution(edge) == -1 else edge

                old_color = state.node_to_color[iedge[0]]  # find the district that the original belongs to
                new_color = state.node_to_color[iedge[1]]
                proposed_smaller = state.graph.subgraph(
                    [i for i in state.color_to_node[old_color] if i != iedge[0]])  # the graph that lost a node
                # proposed_bigger = state.graph.subgraph(state.color_to_node[iedge[1]] + [iedge[0]])
                if len(proposed_smaller) and nx.is_connected(
                        proposed_smaller):  # len(proposed_smaller) checks we didn't eliminate the last node in district
                    continue  # can't disconnect districts

                # TODO population constraint enforcement - optional, min/max check
                # TODO constraint on compactness -
                # TODO constraint on traversals - figure out later

                score = self.score_proposal(iedge[0], old_color, state.node_to_color[iedge[1]],
                                            state)  # smaller score is better
                proposals[(iedge[0],
                           new_color)] += score  # in case multiple valid contested edges for one node, we want to sum them up

        if not len(proposals):
            return None, 0.0  # this happens, just involve
        else:
            prop_sum = sum(proposals.values())
            # pick a proposal
            prop_keys_list = list(proposals.keys())

            idx = np.random.choice(range(len(prop_keys_list)), p=[i / prop_sum for i in proposals.values()])
            choice = prop_keys_list[idx]
            return choice, proposals[choice]  # proposal, score

    def score_proposal(self, node_id, old_color, new_color, state):
        # scores based on summed L2 norm of population (so an even spread will minimize)
        # equivalent to comparing

        current_score = sum([i ** 2 for i in state.stat_tally[self.statistic].values()])
        sum_smaller = sum([state.node_data[i][self.statistic] for i in state.color_to_node[old_color] if i != node_id])
        sum_larger = state.stat_tally[self.statistic][new_color] + state.node_data[node_id][self.statistic]
        new_score = current_score - state.stat_tally[self.statistic][new_color] ** 2 - state.stat_tally[self.statistic][
            old_color] ** 2 + sum_smaller ** 2 + sum_larger ** 2
        print(np.sqrt(new_score) - np.sqrt(current_score))
        return np.exp((np.sqrt(new_score) - np.sqrt(current_score)) / 5000)

        # TODO this feels clumsy, mostly just for demonstration purposes

    def handle_rejection(self, prop, state):
        state.involution *= -1
        # TODO refactor state_log to allow different types of events


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
            districts_to_update = {district_id for node_id, district_id in
                                   state.state_log[state.district_to_com_updated:]}
            for district_id in districts_to_update:
                state.district_to_com[district_id] = compute_com(state, district_id)
        return func(state, *args, **kwargs)

    return inner


