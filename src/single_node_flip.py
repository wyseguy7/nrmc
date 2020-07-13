import random
import copy

from .core import MetropolisProcess, TemperedProposalMixin
from .state import update_contested_edges, check_population, simply_connected
from .scores import cut_length_score


class SingleNodeFlip(MetropolisProcess):
    pass

class SingleNodeFlipTempered(TemperedProposalMixin, SingleNodeFlip):
    pass
