import copy
import numpy as np

from .core import TemperedProposalMixin, compute_dot_product
from .state import update_center_of_mass, update_perimeter_aggressive, update_population, calculate_com_one_step
from .scores import compactness_score, population_balance_score


class CenterOfMassFlow(TemperedProposalMixin):
    def __init__(self, *args, weight_attribute=None, center=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_attribute = weight_attribute


        # produce a center based on weight attribute
        if center is None:
            center = np.array([0,0], dtype='d')
            for i in range(2): # will this always be R2?
                center[i] = sum(nodedata['Centroid'][i]*
                                (nodedata[weight_attribute] if weight_attribute is not None else 1)
                                for node, nodedata in self.state.graph.nodes.items())

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

        # these are all simple objects, so copy is safe - how expensive is it though?
        self.state.contested_nodes = copy.copy(self._proposal_state.contested_nodes)
        self.state.contested_edges = copy.copy(self._proposal_state.contested_edges)
        self.state.com_centroid = copy.copy(self._proposal_state.com_centroid)
        self.state.com_total_weight = copy.copy(self._proposal_state.com_total_weight)
        self.state.articulation_points = copy.copy(self._proposal_state.articulation_points)

        self.state.articulation_points_updated += 1
        self.state.com_updated += 1
        self.state.contested_edges_updated +=1


    def proposal_filter(self, state, proposals):

        update_center_of_mass(state)
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

class CenterOfMassLazyInvolution(CenterOfMassFlow):
    def __init__(self, *args, involution_rate=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.involution_rate = involution_rate

    def perform_involution(self):
        # TODO rng here
        if np.random.uniform(size=1) < self.involution_rate:
            super().perform_involution() # involve



class CenterOfMassIsoparametricPopulation(CenterOfMassFlow):

    def __init__(self, *args, compactness_weight=1., population_weight=1., **kwargs):
        super().__init__(*args, **kwargs)
        self.compactness_weight = compactness_weight
        self.population_weight = population_weight

        # self.pop_score_log = [copy.deepcopy(self.pop_score)]
        self.compactness_score_log = []


    def get_proposals(self, state):
        # need to make sure these are updated
        update_perimeter_aggressive(state)
        update_population(state)
        return super().get_proposals(state)

    def score_proposal(self, node_id, old_color, new_color, state):
        return (self.compactness_weight * compactness_score(state, (node_id, old_color, new_color))
        + self.population_weight * (population_balance_score(state, (node_id, old_color, new_color))-state.population_deviation))
