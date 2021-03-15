import copy
from .core import MetropolisProcess, TemperedProposalMixin
from numpy.random import gamma
import numpy as np

class SingleNodeFlip(MetropolisProcess):
    pass

class SingleNodeFlipTempered(TemperedProposalMixin, SingleNodeFlip):
    pass


class SingleNodeFlipGibbs(SingleNodeFlip):


    def __init__(self, *args, var_a=10., var_b=10., **kwargs):
        super().__init__(*args, **kwargs)
        self.var_a = var_a
        self.var_b = var_b


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
        phi_new = gamma(self.var_a+self.state.N, 2/(self.var_b - self.state.likelihood + self.state.yty*self.state.p)) #
        self.phi_log.append(phi_new)
        self.state.phi = phi_new

        self.likelihood_log.append(copy.deepcopy(self.state.likelihood))

