import os
import pickle
import uuid
import copy
import numpy as np
import random
import itertools
import networkx as nx

from .state import update_boundary_nodes, update_contested_edges, update_population, check_population, simply_connected, connected_breadth_first, update_center_of_mass

from .scores import cut_length_score

ROT_MATRIX = np.matrix([[0, -1], [1, 0]])
CENTROID_DIM_LENGTH = 2 # TODO do we need a settings.py file?
exp = lambda x: np.exp(min(x, 700)) # avoid overflow


try:
    from biconnected import biconnected_dfs, dot_product
    cython_biconnected = True
except ImportError:
    print("No Cython for you!")
    cython_biconnected = False



class MetropolisProcess(object):

    def __init__(self, state, beta=1, measure_beta=1, log_com = True):
        self._initial_state = copy.deepcopy(state)  # save for later
        self.state = state
        self.score_log = [] # for debugging
        self.beta = beta
        self.measure_beta = measure_beta
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
        update_population(state)
        # pop_check_failed = set()

        for node_id, old_color, new_color in proposals:
            # if old_color in pop_check_failed:
            #     continue

            if self.state.minimum_population is not None and not check_population(state, node_id, old_color, self.state.minimum_population):
                # pop_check_failed.add(old_color)  # can't do this anymore
                continue

            if node_id in state.articulation_points[old_color] or not simply_connected(state, node_id, old_color, new_color) \
                    or not self.proposal_check(state, (node_id, old_color, new_color)):
                continue # will disconnect the graph

            yield (node_id, old_color, new_color)


    def proposal_checks(self, state, proposal):
        # deprecated
        node_id, new_color, old_color = proposal # should really make a named tuple for this, tired of unpacking

        if self.state.minimum_population is not None and not check_population(state, node_id, old_color, self.state.minimum_population):
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