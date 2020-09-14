import numpy as np


from .core import MetropolisProcess, TemperedProposalMixin, compute_dot_product
from .updaters import update_contested_edges


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




