import networkx as nx
from collections import defaultdict
import numpy as np
import copy
import random


from .core import MetropolisProcess, TemperedProposalMixin, compute_dot_product, exp
from .state import update_contested_edges, simply_connected, check_population


class PrecintFlow(MetropolisProcess):
    # TODO perhaps stat_tally should go in here instead

    # make proposals on what maximizes the L2 norm of a statistic
    def __init__(self, *args, statistic='population', center=(0, 0), **kwargs):
        super().__init__(*args, **kwargs)
        self.statistic = statistic
        self.state.involution = 1  # also do involutions
        self.center = np.array(center, dtype='d') # guarantee a numpy array here
        self.make_involution_lookup_naive()  # instantiate the involution lookup
        # self.ideal_statistic = sum([node[self.statistic] for node in self.state.graph.nodes().values()])/len(self.state.graph.nodes())
        # self.ideal_statistic = sum([self.state.node_data[node_id][self.statistic]
        #                             for node_id in self.state.node_to_color.keys()]) / len(self.state.color_to_node)
        # assumes balance
        # self.score_log = []


        # self._proposal_state = copy.deepcopy(self.state)
        # self._proposal_state.involution *= -1 # should be opposite of original state
        # self._proposal_state.log_contested_edges = False # don't waste time logging this

    def make_involution_lookup_naive(self):
        self.involution_lookup = dict()
        for edge in self.state.graph.edges:  # these edges are presumed undirected, we have to sort out directionality
            n1 = self.state.graph.nodes()[edge[0]]
            n2 = self.state.graph.nodes()[edge[1]]
            # should this be greater or less than zero?
            if compute_dot_product(np.array(n1['Centroid'], dtype='d'),
                                   np.array(n2['Centroid'], dtype='d'), center=self.center) >= 0:
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


    def get_directed_edges(self, state):
        update_contested_edges(state)
        # does NOT do any checks for connected components, etc
        return [(edge[1], edge[0]) if self.get_involution(state, edge) == -1 else edge for edge in state.contested_edges]


    # def proposal(self, state):
    #     # pick a conflicted edge at random. since this is symmetric under involution, no need to compute Q(z,z')/Q(z',z).
    #
    #     update_contested_edges(state)
    #
    #     # TODO do we need to find Q(x,x') here since we're restricting to states that don't disconnect the graph?
    #     # don't think so
    #     edges = random.sample(state.contested_edges,
    #                           len(state.contested_edges))  # TODO this is currently an O(n) operation technically,
    #
    #     for edge in edges:
    #         iedge = (edge[1], edge[0]) if self.get_involution(state, edge) == -1 else edge
    #         old_color, new_color = self.state.node_to_color[iedge[0]], self.state.node_to_color[iedge[1]]  # for clarity
    #         proposed_smaller = self.state.graph.subgraph(
    #             [i for i in self.state.color_to_node[old_color] if i != iedge[0]])  # the graph that lost a node
    #         if not len(proposed_smaller) or not nx.is_connected(
    #                 proposed_smaller):  # len(proposed_smaller) checks we didn't eliminate the last node in district
    #             continue  # can't disconnect districts or eliminate districts entirely
    #
    #         score = self.score_proposal(iedge[0], old_color, new_color, self.state)
    #         return (iedge[0], new_color), score
    #     raise RuntimeError("Exceeded tries to find a proposal")


class PrecintFlowTempered(PrecintFlow, TemperedProposalMixin):
    pass # woohoo! no code to write here


    # def proposal(self, state):
    #
    #     # get all the proposals, with their associated probabilities
    #     proposals = self.get_proposals(state, filter_connected=False)
    #     # pick a proposal
    #     try:
    #         proposal, score = self.pick_proposal(proposals)
    #     except ValueError:
    #         # state.involution *=-1
    #         # self._proposal_state.involution *= -1
    #         # return self.proposal(state) # if we can't find a valid proposal
    #         return (None, None, None), 0 # we couldn't find a valid proposal, this is guaranteed to cause involution
    #
    #     node_id, old_color, new_color = proposal
    #     self._proposal_state.handle_move(proposal)
    #     reverse_proposals = self.get_proposals(self._proposal_state, filter_connected=True)
    #     reverse_proposal_probs = {k:self.score_to_proposal_prob(v) for k,v in reverse_proposals.items()}
    #     proposal_probs = {k: self.score_to_proposal_prob(v) for k,v in proposals.items()}
    #
    #     try:
    #         q_prime = reverse_proposals[(node_id, new_color, old_color)]/sum(reverse_proposal_probs.values())
    #         q = proposals[proposal]/sum(proposal_probs.values())
    #         return proposal, q_prime/q*exp(-score*self.beta*self.measure_beta) # TODO should this be a different func?
    #     except KeyError:
    #         return proposal, 0 # sometimes reverse is not contained in proposals list


    # def handle_rejection(self, prop, state):
    #     super().handle_rejection(prop, state)
    #     if prop[0] is not None: # we couldn't find a valid proposal
    #         self._proposal_state.handle_move((prop[0], prop[2], prop[1])) # flip back to old color
    #     self._proposal_state.involution *= -1

    # def pick_proposal(self, proposals):
    #
    #     # threshold = np.random.random()
    #     # keys = sorted(proposals.keys())
    #     counter = 0
    #     while True:
    #         proposal = random.choices(list(proposals.keys()), weights=self.score_to_proposal_prob(proposals.values()), k=1)[0]
    #         if self.check_connected(self.state, proposal[0], proposal[1], proposal[2]):
    #             return proposal, proposals[proposal]
    #         else:
    #             counter +=1
    #             proposals[proposal] = 0 # this wasn't connected, so we can't pick it
    #             if counter >= len(proposals):
    #                 # return (None, None, None), 0 # not actually a valid proposal, guaranteed to result in involution
    #                 raise ValueError("Could not find connected proposal")
    #
    #             proposals[proposal] = 0 # probability is zero if it's not connected

    # def get_proposals(self, state=None, filter_connected=False):
    #
    #     update_contested_edges(state)
    #     # update_perimeter_aggressive(state)
    #     proposals = defaultdict(int)
    #
    #     directed_edges = self.get_directed_edges(state)
    #     for iedge in directed_edges:
    #
    #         old_color = state.node_to_color[iedge[0]]  # find the district that the original belongs to
    #         new_color = state.node_to_color[iedge[1]]
    #
    #         if not check_population(state, old_color, minimum=500):
    #             continue # ensure that population of either does not drop below 500
    #             # TODO parameterize this
    #
    #         if (iedge[0], old_color, new_color) in proposals:
    #             continue # we've already scored this proposal, no need to redo
    #
    #         if filter_connected and not self.check_connected(state, iedge[0], old_color, new_color):
    #              continue
    #
    #         score = self.score_proposal(iedge[0], old_color, new_color, state)  # smaller score is better
    #         proposals[(iedge[0], old_color, new_color)] = score
    #
    #     return proposals

