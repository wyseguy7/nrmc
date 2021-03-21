import copy
from .core import MetropolisProcess, TemperedProposalMixin
from numpy.random import gamma, normal
from numpy.linalg import cholesky
import numpy as np

class SingleNodeFlip(MetropolisProcess):
    pass

class SingleNodeFlipTempered(TemperedProposalMixin, SingleNodeFlip):
    pass


