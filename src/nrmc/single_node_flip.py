import copy
from .core import MetropolisProcess, TemperedProposalMixin
from numpy.random import gamma

class SingleNodeFlip(MetropolisProcess):
    pass

class SingleNodeFlipTempered(TemperedProposalMixin, SingleNodeFlip):
    pass


class SingleNodeFlipGibbs(SingleNodeFlip):


    def __init__(self, *args, var_a=1., var_b=1., **kwargs):
        super().__init__(*args, **kwargs)
        self.var_a = var_a
        self.var_b = var_b

        self.state.phi = gamma(self.var_a, 1/self.var_b)

        self.phi_log = []
        self.likelihood_log = []


    def handle_acceptance(self, prop, state):
        super().handle_acceptance(prop, state)
        self.state.likelihood = self.state.prop_likelihood
        self.state.inv = self.state.prop_inv
        self.state.W  = self.state.prop_W
        self.state.inv_det_log = self.state.prop_det_log


    def step(self):

        super().step()
        phi_new = gamma(self.var_a, 1/(self.var_b + self.state.likelihood + self.state.yty)) #
        self.phi_log.append(phi_new)
        self.state.phi = phi_new

        self.likelihood_log.append(copy.deepcopy(self.state.likelihood))

