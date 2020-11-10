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


    # def get_directed_edges(self, state):
    #
    #     raise ValueError("")
    #
    #     self.update_boundary_scores(state)
    #     # want to randomly select but weight based on acceptance probability
    #
    #     if self.boundary is None: # only pick if not
    #         boundary = random.choices(state.boundary_totals_dict.keys(), weights=state.boundary_totals_dict.values())
    #         # we don't need to check for emptiness because now we'll just skip it
    #
    #         if self.state.involution_state[boundary] == 1:
    #             old_color, new_color = boundary
    #
    #         else:
    #             new_color, old_color = boundary
    #
    #         self.boundary = (old_color, new_color) # always stored in the direction of flow
    #
    #     else:
    #         # boundary has already been picked, meaning we are the reversed proposal - right?
    #
    #         new_color, old_color = self.boundary
    #         boundary = (min(old_color, new_color), max(old_color, new_color)) # correct sorted order, for searching
    #
    #     # TODO should we also store this for the reverse proposal?
    #
    #     for edge in state.district_boundary[boundary]:
    #         if state.node_to_color[edge[0]] == old_color:
    #             yield edge
    #         else:
    #             yield (edge[1], edge[0])


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


        for move in state.move_log:
            if move is not None:
                node_id, old_color, new_color = move

                for boundary in state.score_dict:
                    if old_color in boundary or new_color in boundary:
                        state.score_dict[boundary] = self.rescore_boundary(state, boundary)
                        state.boundary_totals_dict[boundary] = sum(self.score_to_proposal_prob(score) for score in state.score_dict[boundary].values())


    # def handle_rejection(self, prop, state):
    #     super().handle_rejection(prop, state)
    #
    #
    #     self.update_boundary_scores(self.state)
        # node_id, old_color, new_color = prop
        # boundary = (min(old_color, new_color), max(old_color, new_color))
        # self.state.score_dict[boundary] = self.rescore_boundary(self.state, boundary)
        # self._proposal_state.score_dict[boundary] = self.rescore_boundary(self._proposal_state, boundary)
        #
        # self.state.boundary_totals_dict[boundary] = sum(self.score_to_proposal_prob(score) for score in self.state.score_dict[boundary].values())
        # self._proposal_state.boundary_totals_dict[boundary] = sum(self.score_to_proposal_prob(score) for score in self._proposal_state.score_dict[boundary].values())

        # self._proposal_state.boundary_totals_dict = copy.copy(self.state.boundary_totals_dict)
        # self._proposal_state.score_dict = copy.copy(self.state.score_dict)


    def boundary_scores_naive(self, state):

        update_district_boundary(state)
        score_dict = collections.defaultdict()
        for boundary in state.district_boundary:
            score_dict[boundary] = self.rescore_boundary(state, boundary)
            # score_totals_dict[boundary] = sum(self.score_to_proposal_prob(score) for score in score_dict[boundary].values())

        return score_dict

    # def handle_acceptance(self, prop, state):
    #     super().handle_acceptance(prop, state)
    #     # these are all simple objects, so copy is safe - how expensive is it though?
    #
    #     self.state.district_boundary = copy.copy(self._proposal_state.district_boundary)
    #     self.state.district_boundary_updated += 1
    #
    #     # self.state.score_dict = copy.copy(self._proposal_state.score_dict)
    #     # self.state.boundary_totals_dict = copy.copy(self._proposal_state.boundary_totals_dict)
    #
    #     # self.state.boundary_score_updated += 1


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

class DistrictToDistrictTempered(TemperedProposalMixin):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.boundary = None

        # assign initial involution state - should this be specifiable by user?
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
        # self._proposal_state.involution_state = self.state.involution_state


    def involve_state(self, state):
        state.involution_state[min(self.boundary), max(self.boundary)]*=-1

    # def handle_rejection(self, prop, state):
    #     super().handle_rejection(prop, state)
    #     self.involution_state[min(prop[1], prop[2]), max(prop[1], prop[2])]*= -1 # flip the appropriate involution state

    def handle_acceptance(self, prop, state):
        super().handle_acceptance(prop, state)
        # these are all simple objects, so copy is safe - how expensive is it though?

        self.state.district_boundary = copy.copy(self._proposal_state.district_boundary)
        self.state.district_boundary_updated += 1


    def step(self):
        self.boundary = None
        super().step()


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
        if self.boundary is None: # pick the involution
            counter = 0
            while True:
                boundary = random.choices(list(state.district_boundary.keys()))[0]
                if state.district_boundary[boundary]: # if this isn't an empty list
                    self.boundary = boundary
                    break
                counter +=1
                if counter > 100:
                    raise ValueError("Didn't select a valid boundary within valid time")
        else:
            boundary = self.boundary

        if state.involution_state[boundary] == 1:
            old_color, new_color = boundary
        else:
            # state = -1
            new_color, old_color = boundary

        # self.boundary = (old_color, new_color) # always in the direction of flow

        for edge in state.district_boundary[boundary]:
            if state.node_to_color[edge[0]] == old_color:
                yield edge
            else:
                yield (edge[1], edge[0])




class DistrictToDistrictLazyInvolution(DistrictToDistrictFlow):
    def __init__(self, *args, involution_rate=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.involution_rate = involution_rate


    def get_involution_probability(self):
        props = self.get_proposals(self.state)
        if props:
            involution_prob = sum(min(1, self.score_to_prob(score)) for score in props.values())/len(props)
        else:
            involution_prob = 1. # no valid proposals means a guaranteed involution

        # involve to collect reverse probabilities
        super().perform_involution()
        rev_props = self.get_proposals(self.state)
        if rev_props:
            reverse_involution_prob = sum(min(1, self.score_to_prob(score)) for score in rev_props.values())/len(rev_props)
        else:
            reverse_involution_prob = 1.
        # assumes that scores are reversable
        super().perform_involution() # undo the involution - major programmer guilt but I think it's fine

        return min(involution_prob, reverse_involution_prob)


    def perform_involution(self):
        # TODO rng here

        involution_prob = self.get_involution_probability()
        if np.random.uniform(size=1) > involution_prob:
            super().perform_involution() # involve

