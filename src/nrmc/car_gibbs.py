import numpy as np
from numpy.linalg import cholesky
from numpy.random import gamma, normal
import copy
import random

from .core import MetropolisProcess, TemperedProposalMixin
from .scores import car_model_updated, approx_car_score
from .center_of_mass import update_perimeter_and_area, calculate_com_one_step, update_center_of_mass,  compute_dot_product


class GibbsMixing(MetropolisProcess):


    def __init__(self, *args, var_a=10., var_b=10., **kwargs):
        super().__init__(*args, **kwargs)
        self.var_a = var_a
        self.var_b = var_b


        self.phi_log = []
        self.likelihood_log = []
        self.score_tracker_log = []


        # this is a horrible pattern for which I am sorry
        self.approx_score_list = copy.deepcopy(self.score_list)
        for i in range(len(self.approx_score_list)):
            if self.approx_score_list[i][1] is car_model_updated: # TODO keep this updated
                # rip out precise score, replace with approx score
                self.approx_score_list[i] = (self.approx_score_list[i][0], approx_car_score)


    # def handle_acceptance(self, prop, state):
    #     super().handle_acceptance(prop, state)
    #     # self.state.likelihood = self.state.prop_likelihood
    #     # self.state.inv = self.state.prop_inv
    #     # self.state.W  = self.state.prop_W
    #     # self.state.inv_det_log = self.state.prop_det_log
    #     # self.state.U = self.state.prop_U


    def step(self):

        super().step()
        phi_new = gamma(self.var_a+self.state.N, self.state.p/(self.var_b - self.state.likelihood + self.state.yty*self.state.p)) #
        self.phi_log.append(phi_new)
        self.state.phi = phi_new

        self.likelihood_log.append(copy.deepcopy(self.state.likelihood)) # do we need to store this?

    @property
    def beta_mle(self):
        return self.state.inv @ self.state.xty

    def sample_beta(self):

        normal_noise = normal(size=self.state.p)
        beta_chol = cholesky(self.state.inv)


        return self.beta_mle + beta_chol @ normal_noise

        # given the current state, estimate


class TemperedGibbs(TemperedProposalMixin, GibbsMixing):


    def score_proposal_approx(self, node_id, old_color, new_color, state):
        score_sum = 0.
        for score_weight, score_func  in self.approx_score_list:
            score_sum += score_weight * score_func(state, (node_id, old_color, new_color))
        return score_sum

    def handle_acceptance(self, prop, state):
        super().handle_acceptance(prop, state)
        self.state.likelihood = copy.deepcopy(self._proposal_state.likelihood)
        self.state.inv = copy.deepcopy(self._proposal_state.inv)
        self.state.W  = copy.deepcopy(self._proposal_state.W)
        self.state.inv_det_log = copy.deepcopy(self._proposal_state.inv_det_log)
        self.state.U = copy.deepcopy(self._proposal_state.U)
        self.state.car_updated = self.state.iteration



    def proposal(self, state):
        # TODO this is common with SingleNodeFlipTempered, perhaps split this out
        proposals = self.get_proposals(self.state)
        scored_proposals = {(node_id, old_color, new_color):self.score_proposal_approx(node_id, old_color, new_color, state)
                            for node_id, old_color, new_color in proposals}

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
        reverse_scored_proposals = {(node_id, old_color, new_color):self.score_proposal_approx(node_id, old_color, new_color, state)
                            for node_id, old_color, new_color in reverse_proposals}
        reverse_proposal_probs = {k: self.score_to_proposal_prob(v) for k, v in reverse_scored_proposals.items()}
        try:
            q = reverse_proposal_probs[(proposal[0], proposal[2], proposal[1])] / sum(
                reverse_proposal_probs.values())
            q_prime = proposal_probs[proposal] / sum(proposal_probs.values())
            node_id, old_color, new_color = proposal
            exact_score = self.score_proposal(node_id, old_color, new_color, state)

            approx_score = scored_proposals[proposal]
            self.score_tracker_log.append((exact_score, approx_score))

            prob = self.score_to_prob(exact_score) * q / q_prime
            return proposal, prob
        except KeyError:
            self.no_reverse_prob_counter += 1
            return proposal, 0  # this happens sometimes but probably shouldn't for single node flip


class CenterOfMassFlowGibbs(TemperedGibbs):
    def __init__(self, *args, weight_attribute=None, center=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_attribute = weight_attribute


        # produce a center based on weight attribute
        if center is None:
            center = np.array([0,0], dtype='d')
            for i in range(2): # will this always be R2?
                center[i] = sum(nodedata['Centroid'][i]*
                                (nodedata[weight_attribute] if weight_attribute is not None else 1)
                                for node, nodedata in self.state.graph.nodes.items())/len(self.state.graph.nodes())

        else:
            center = np.array(center, dtype='d') # guarantee a numpy array

        self.center = center

        for state in (self.state, self._proposal_state):
            for node_id in state.graph.nodes():
                # TODO parameterize 'Centroid' instead of hardcoding
                # we need to guarantee these are numpy arrays up front
                state.graph.nodes()[node_id]['Centroid'] = np.array(state.graph.nodes()[node_id]['Centroid'])

    def handle_rejection(self, prop, state):
        # TODO need to make this cleaner

        super().handle_rejection(prop, state)
        # print("Involution on step {}".format(state.iteration))
        # state.involution *= -1

    def handle_acceptance(self, prop, state):
        super().handle_acceptance(prop, state)

        self.state.com_centroid = copy.copy(self._proposal_state.com_centroid)
        self.state.com_total_weight = copy.copy(self._proposal_state.com_total_weight)
        self.state.com_updated += 1


    def proposal_filter(self, state, proposals):

        update_center_of_mass(state, self.weight_attribute)
        # stack this generator on top of the earlier one
        for proposal in super().proposal_filter(state, proposals):
            # filter out any proposals that don't meet the criterion
            node_id, old_color, new_color = proposal
            (centroid_new_color, centroid_old_color,
             total_weight_new_color, total_weight_old_color) = calculate_com_one_step(
                state, (node_id, old_color, new_color), self.weight_attribute)

            try:
                dp_old = compute_dot_product(state.com_centroid[old_color], centroid_old_color, self.center)
                dp_new = compute_dot_product(state.com_centroid[new_color], centroid_new_color, self.center)


                if (dp_old + dp_new)*state.involution > 0:
                    yield proposal

            except ZeroDivisionError:
                yield proposal # rare event, we want to yield


