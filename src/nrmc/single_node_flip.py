from .core import MetropolisProcess, TemperedProposalMixin
import random
import collections

class SingleNodeFlip(MetropolisProcess):
    def get_proposals(self, state):
        # produces a mapping {proposal: score} for each edge in get_directed_edges

        proposals = collections.defaultdict(int)
        # scored_proposals = {}

        for updater in self.score_updaters:
            updater(state)

        for node_id, neighbor in self.get_directed_edges(state):
            old_color = state.node_to_color[node_id]
            new_color = state.node_to_color[neighbor]
            proposals[(node_id, old_color, new_color)] += 1
        # only score proposal if it passes the filter
        return {proposal: proposals[proposal] for proposal in self.proposal_filter(state, proposals.keys())}

        # return scored_proposals


    def proposal(self, state):
        # picks a proposal randomly without any weighting
        proposals = self.get_proposals(self.state) # this is a set now
        # score = self.score_proposal()
        proposal = random.choices(list(proposals.keys()), weights=list(proposals.values()))[0]
        node_id, old_color, new_color = proposal
        prob = self.score_to_prob(self.score_proposal(node_id, old_color, new_color, state))  # should be totally unweighted here
        return proposal, prob


class SingleNodeFlipTempered(TemperedProposalMixin, MetropolisProcess):
    pass
