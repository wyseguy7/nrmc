import random
import numpy as np
import copy
import collections

from .core import MetropolisProcess, TemperedProposalMixin
from .updaters import update_contested_edges, update_district_boundary



class DistrictToDistrictFixed(TemperedProposalMixin):


    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        involution_state = dict()
        for district_id in self.state.color_to_node.keys():
            for other_district_id in self.state.color_to_node.keys():
                if district_id < other_district_id:
                    involution_state[(district_id, other_district_id)] = random.choices([-1, 1])[0]
        self.state.involution_state = involution_state
        self.log_com = False  # TODO repair the issue here

        # new to tempered_side
        self._proposal_state.involution_state = dict()
        for key in self.state.involution_state:
            self._proposal_state.involution_state[key] = self.state.involution_state[key] * -1


    def step(self):
        self.boundary = None
        super().step()

    def involve_state(self, state):
        state.involution_state[min(self.boundary), max(self.boundary)]*=-1


    def proposal(self, state):
        proposal, prob = super().proposal(state)
        # add in edge reweighting here

        boundary = (min(self.boundary), max(self.boundary)) # put boundary in correct order for lookup purposes
        odds_picking_e = self.state.boundary_totals_dict[boundary]/sum(self.state.boundary_totals_dict.values())
        rev_odds_picking_e = self._proposal_state.boundary_totals_dict[boundary]/sum(self._proposal_state.boundary_totals_dict.values())

        # TODO be careful of numerical stability here if these ratios become small - its clearer this way but less stable
        return proposal, prob*rev_odds_picking_e/odds_picking_e

    def get_proposals(self, state):

        update_district_boundary(state)
        self.update_boundary_scores(state) # make sure this is updated

        # pick a district edge
        if self.boundary is None: # only pick if not
            boundary = random.choices(list(state.boundary_totals_dict.keys()), weights=list(state.boundary_totals_dict.values()))[0]
            # we don't need to check for emptiness because now we'll just skip it

            if state.involution_state[boundary] == 1:
                old_color, new_color = boundary

            else:
                # state = -1
                new_color, old_color = boundary

            self.boundary = (old_color, new_color) # always stored in the direction of flow

        else:
            # boundary has already been picked, meaning we are the reversed proposal - right?

            new_color, old_color = self.boundary
            boundary = (min(old_color, new_color), max(old_color, new_color)) # correct sorted order, for searching

        # return proposal based on pre-computed results
        # TODO should we also store this for the reverse proposal?
        return {prop:score  for prop, score in state.score_dict[boundary].items() if prop[1]==old_color}
        # these are pre-computed so we will be fine, but need to filter now for involution. annoyingly this is O(n), can fix?


    def rescore_boundary(self, state, boundary):

        # guarantee updated
        for updater in self.score_updaters:
            updater(state)

        # if state.involution_state[boundary] == 1:
        #     old_color, new_color = boundary
        # else:
        #     new_color, old_color = boundary

        proposals = set()
        color1, color2 = boundary

        for nodes in state.district_boundary[boundary]:
            for node_id in nodes: # iterate through both

                if state.node_to_color[node_id] == color1:
                    proposal = (node_id, color1, color2)
                else:
                    proposal = (node_id, color2, color1)

                proposals.add(proposal)

        proposals = self.proposal_filter(state, proposals)

            # my_dict[proposal] = self.score_proposal(proposal[0], old_color, new_color, state)
        return {proposal: self.score_proposal(proposal[0], proposal[1], proposal[2], state) for proposal in proposals}


    def update_boundary_scores(self, state):
        update_district_boundary(state) # ensure this is updated first

        if not hasattr(state, 'score_dict'):
            state.score_dict = self.boundary_scores_naive(state)
            state.boundary_totals_dict = {boundary: sum(self.score_to_proposal_prob(score)
                                                        for score in state.score_dict[boundary].values()) for boundary in state.district_boundary}
            state.boundary_score_updated = state.iteration


        for move in state.move_log[state.boundary_score_updated:]:
            if move is not None:
                node_id, old_color, new_color = move

                for boundary in state.score_dict:
                    if old_color in boundary or new_color in boundary:
                        state.score_dict[boundary] = self.rescore_boundary(state, boundary)
                        state.boundary_totals_dict[boundary] = sum(self.score_to_proposal_prob(score) for score in state.score_dict[boundary].values())



        state.boundary_score_updated = state.iteration


    def boundary_scores_naive(self, state):

        update_district_boundary(state)
        score_dict = collections.defaultdict()
        for boundary in state.district_boundary:
            score_dict[boundary] = self.rescore_boundary(state, boundary)
            # score_totals_dict[boundary] = sum(self.score_to_proposal_prob(score) for score in score_dict[boundary].values())

        return score_dict

    def handle_acceptance(self, prop, state):
        super().handle_acceptance(prop, state)

        self.state.boundary_totals_dict = copy.copy(self._proposal_state.boundary_totals_dict)
        self.state.score_dict = copy.copy(self._proposal_state.score_dict)
        self.state.boundary_score_updated = self.state.iteration # mark as up-to-date


