from .core import MetropolisProcess, TemperedProposalMixin


class SingleNodeFlip(MetropolisProcess):
    pass

class SingleNodeFlipTempered(TemperedProposalMixin, SingleNodeFlip):
    pass
