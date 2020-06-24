import random
import copy

from .core import MetropolisProcess
from .state import update_contested_edges, check_population, simply_connected
from .scores import cut_length_score


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

                if not check_population(state, old_color, minimum=self.minimum_population): # TODO error here
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
