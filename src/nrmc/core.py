import os
import pickle
import uuid
import copy
import random
import itertools
import json

import numpy as np
import networkx as nx

from .state import connected_breadth_first, State, np_to_native
from .constraints import simply_connected
from .updaters import update_center_of_mass, update_contested_edges, update_perimeter_and_area, \
    update_population, check_population, update_boundary_nodes, update_parcellation
from .scores import cut_length_score, population_balance_score, population_balance_sq_score, compactness_score, gml_score

ROT_MATRIX = np.matrix([[0, -1], [1, 0]])
exp = lambda x: np.exp(min(x, 700)) # avoid overflow


try:
    from .biconnected import biconnected_dfs, dot_product, calculate_com_inner, PerimeterComputer
    cython_biconnected = True
except ImportError:
    print("No Cython for you!")
    cython_biconnected = False


score_lookup = {'cut_length': cut_length_score,
                'compactness': compactness_score,
                'population_balance': population_balance_sq_score,
                'gsl': gml_score
                } # TODO rip out population_balance calculation from updater

score_updaters = {'cut_length': [],
                  'compactness': [update_perimeter_and_area],
                  'population_balance': [update_population],
                  'gsl': [update_parcellation]
  }

class ProcessEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (MetropolisProcess, State)):
            return o._json_dict
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        else:
            return super().default(o)


class MetropolisProcess(object):

    def __init__(self, state, beta=1, measure_beta=1, log_com = False, folder_path = '/gtmp/etw16/runs/',
                 score_funcs=('cut_length',), score_weights=(1.,), unique_id=None, **kwargs):

        self.score_list = []
        self.score_updaters = []
        self.score_funcs = score_funcs # we need to store this to make json-serializable

        for score, score_weight in zip(score_funcs, score_weights):
            self.score_list.append((score_weight, score_lookup[score]))
            for updater in score_updaters[score]: # these are functions that need to run before computing a particular score
                self.score_updaters.append(updater)

        self.state = state
        self.score_log = [] # for debugging
        self.beta = beta
        self.measure_beta = measure_beta
        if not hasattr(state, 'involution'):
            state.involution = 1 # unless this comes from elsewhere, it should have an involution

        self.log_com = log_com
        if log_com:
            self.com_log = []
        self.folder_path = folder_path

        if unique_id is None:
            self.unique_id = uuid.uuid4().hex[:6]
            self.make_sandbox()
        else:
            self.unique_id = unique_id # assumes sandbox already created

        if cython_biconnected and 'compactness' in score_funcs:

            border_length_lookup = {MetropolisProcess.pack_int(*edge):self.state.graph.edges()[edge]['border_length'] for edge in self.state.graph.edges()}
            border_length_lookup.update({MetropolisProcess.pack_int(edge[1], edge[0]):self.state.graph.edges()[edge]['border_length'] for edge in self.state.graph.edges()})
            # we need both directions just to be sure

            external_border_lookup = {node_id: self.state.graph.nodes()[node_id]['external_border'] for node_id in self.state.graph.nodes()}
            adj_mapping_full = {node_id: list(self.state.graph.neighbors(node_id)) for node_id in self.state.graph.nodes()}

            self.state.perimeter_computer = PerimeterComputer(adj_mapping_full,  self.state.color_to_node,
                                                        border_length_lookup, external_border_lookup)
            # self.state.cython_biconnected = True

        self._initial_state = copy.deepcopy(state)  # save for later

        # for k, v in kwargs.items():
        #     setattr(self, k, v) # accept and attach

    # needed for perimeter computer
    @staticmethod
    def pack_int(a, b):
        return (a << 32) | b

    @property
    def run_id(self):
        return "{classname}_{graph_type}_{my_id}".format(classname=self.__class__.__name__,
                                                         my_id= self.unique_id,
                                                         graph_type=self.state.graph_type)


    def make_sandbox(self):
        os.makedirs(os.path.join(self.folder_path, self.run_id), exist_ok=False) # ensure that the folder path exists and is unique

    def save(self):
        with open(os.path.join(self.folder_path, self.run_id, '{}_process.pkl'.format(self.run_id)), mode='wb') as f:
            pickle.dump(self, f)

        with open(os.path.join(self.folder_path, self.run_id, '{}_process.json'.format(self.run_id)), mode='w') as f:
            # json.dump(self.toJson(), f) # boy do I hope this works
            f.write(json.dumps(self, cls=ProcessEncoder, indent=4))


    @property
    def _json_dict(self):
        # write out significant features as needed

        ignore = {"score_updaters", "score_list"}
        custom_dict = {}
        other_dict = {k: v for k,v in self.__dict__.items() if k not in custom_dict and k not in ignore}
        other_dict.update(custom_dict)
        return np_to_native(other_dict)
        # return  json.dumps(other_dict, default=lambda o: o.__dict__) # this should work?

    @classmethod
    def from_json(cls, filepath):
        # TODO extract from json

        with open(filepath) as f:
            js = json.load(f)

        state = State.from_json(js.pop('state'))
        initial_state = js.pop('_initial_state') # we need to override this
        process = cls(state, **js)
        process._initial_state = initial_state # are we going to have an issue with proposal_state?
        return process



    def score_to_prob(self, score):
        return exp(-0.5*self.measure_beta*score)

    def score_to_proposal_prob(self, score):
        return exp(-0.5*score*self.beta)


    def involve_state(self, state):
        state.involution *= -1

    def perform_involution(self):
        self.involve_state(self.state)

    def proposal_filter(self, state, proposals):

        update_boundary_nodes(state)
        update_articulation_points(state)
        update_population(state)
        # pop_check_failed = set()

        for node_id, old_color, new_color in proposals:
            # if old_color in pop_check_failed:
            #     continue

            if self.state.minimum_population is not None and not check_population(state,
                                                                                  node_id, old_color, new_color,
                                                                                  minimum=self.state.minimum_population,
                                                                                  maximum=self.state.maximum_population):
                # pop_check_failed.add(old_color)  # can't do this anymore
                continue

            if node_id in state.articulation_points[old_color] or not simply_connected(state, node_id, old_color, new_color):
                continue # will disconnect the graph

            yield (node_id, old_color, new_color)


    def score_proposal(self, node_id, old_color, new_color, state):
        score_sum = 0.
        for score_weight, score_func  in self.score_list:
            score_sum += score_weight * score_func(state, (node_id, old_color, new_color))
        return score_sum

    # def score_proposal(self, node_id, old_color, new_color, state):
    #     return cut_length_score(state, (node_id, old_color, new_color))

    def get_directed_edges(self, state):
        # default behavior gets all edges in each direction
        update_contested_edges(state)
        return itertools.chain(state.contested_edges, [(edge[1], edge[0]) for edge in state.contested_edges])

    def get_proposals(self, state):
        # produces a mapping {proposal: score} for each edge in get_directed_edges

        proposals = set()
        # scored_proposals = {}

        for updater in self.score_updaters:
            updater(state)


        for node_id, neighbor in self.get_directed_edges(state):
            old_color = state.node_to_color[node_id]
            new_color = state.node_to_color[neighbor]

            proposals.add((node_id, old_color, new_color))

            # if (node_id, old_color, new_color) in proposals:
            #     continue # already scored this proposal

            # if not self.proposal_checks(state, (node_id, old_color, new_color)):
            #     continue # not a valid proposal

            # scored_proposals[(node_id, old_color, new_color)] = self.score_proposal(node_id, old_color, new_color, state)


        # only score proposal if it passes the filter
        return {(node_id, old_color, new_color): self.score_proposal(node_id, old_color, new_color, state)
                for node_id, old_color, new_color in self.proposal_filter(state, proposals)}

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

        self.state.check_connectedness = False  # turn this off - we check it differently here
        self._proposal_state = copy.deepcopy(self.state)

        # if involution is set, make it always the opposite
        self._proposal_state.involution = self.state.involution * -1  # set to opposite of original
        self._proposal_state.log_contested_edges = False  # don't waste time logging this

        self.no_reverse_prob_counter = 0
        self.no_valid_proposal_counter = 0

    def perform_involution(self):
        self.involve_state(self.state)
        self.involve_state(self._proposal_state)


    def handle_acceptance(self, prop, state):
        super().handle_acceptance(prop, state)

        # these are all simple objects, so copy is safe - how expensive is it though?
        self.state.contested_nodes = copy.copy(self._proposal_state.contested_nodes)
        self.state.contested_edges = copy.copy(self._proposal_state.contested_edges)
        self.state.articulation_points = copy.copy(self._proposal_state.articulation_points)
        self.state.population_counter = copy.copy(self._proposal_state.population_counter)
        self.state.population_deviation = copy.copy(self._proposal_state.population_deviation)

        self.state.articulation_points_updated += 1
        self.state.contested_edges_updated +=1
        self.state.population_counter_updated += 1

        if hasattr(state, '_barycenter_proposal'):
            state.barycenter_lookup = state._barycenter_proposal # update from scoring because it's expensive



    def handle_rejection(self, prop, state):
        super().handle_rejection(prop, state)
        if prop[0] is not None:  # ignore if we couldn't find a valid proposal
            self._proposal_state.handle_move((prop[0], prop[2], prop[1]))  # undo earlier move

    def proposal(self, state):
        # TODO this is common with SingleNodeFlipTempered, perhaps split this out
        scored_proposals = self.get_proposals(self.state)
        proposal_probs = {k: self.score_to_proposal_prob(v) for k, v in scored_proposals.items()}

        try:
            proposal = random.choices(list(proposal_probs.keys()),
                                      weights=proposal_probs.values())[0]
        except IndexError:
            self.no_valid_proposal_counter += 1
            return (None, None, None), 0  # we couldn't find a valid proposal, need to involve

        self._proposal_state.handle_move(proposal)
        # if not self.check_connected(state, *proposal) and simply_connected(state, *proposal):
        #     return proposal, 0 # always zero

        reverse_proposals = self.get_proposals(self._proposal_state)
        reverse_proposal_probs = {k: self.score_to_proposal_prob(v) for k, v in reverse_proposals.items()}

        try:
            q = reverse_proposal_probs[(proposal[0], proposal[2], proposal[1])] / sum(
                reverse_proposal_probs.values())
            q_prime = proposal_probs[proposal] / sum(proposal_probs.values())
            prob = self.score_to_prob(scored_proposals[proposal]) * q / q_prime
            return proposal, prob
        except KeyError:
            self.no_reverse_prob_counter += 1
            return proposal, 0  # this happens sometimes but probably shouldn't for single node flip



# TODO we can numba this function if needed
def compute_dot_product(a, b, center=(0, 0), normalize=True):

    if cython_biconnected:
        return dot_product(a, b, center)
    else:

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
            norm_perp = np.linalg.norm(vec_perp_center)
            norm_a_b = np.linalg.norm(vec_a_b)
            if norm_a_b == 0 or norm_perp==0:
                raise ZeroDivisionError("Sorry, you can't do that")
            return dp/norm_perp/norm_a_b

        else:
            return  dp  # dot product


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
