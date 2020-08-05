import random
import copy

from .core import MetropolisProcess, TemperedProposalMixin
from src.nrmc.constraints import simply_connected
from src.nrmc.updaters import update_contested_edges, check_population
from .scores import cut_length_score


class SingleNodeFlip(MetropolisProcess):
    pass

class SingleNodeFlipTempered(TemperedProposalMixin, SingleNodeFlip):
    pass
