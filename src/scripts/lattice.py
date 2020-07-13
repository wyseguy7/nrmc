import sys
import pickle


# sys.path.append('C:\\Users\\wyseg\\nonreversiblecodebase\\src\\')
sys.path.append('/home/grad/etw16/nonreversiblecodebase/') # TODO make this less garbage-y
sys.path.append('/home/grad/etw16/nonreversiblecodebase/src/legacy/') # TODO make this less garbage-y
sys.path.append('/home/grad/etw16/nonreversiblecodebase/src/') # TODO make this less garbage-y


import constructor
import districtingGraph
import initializer
import centerOfMassFlow
import metropolisHastings
from src.state import State


print('initializing run...')
state, args = initializer.setRunParametersFromCommandLine([])
state["step"] = 0
proposal, args = centerOfMassFlow.define(args)
info = args

info = initializer.fillMissingInfoFields(info)
state = initializer.determineStateInfo(state, info)
# state = constructor.contructPlan(state, info)
state = constructor.splitSquareLattice(state, info) ## terrible code; trying to move fast
print('run initialized...')


state_new = State.from_state(state, minimum_population=756)

# load in new 'boundary' attribute to nodes - required to check simply_connectedness
for node in state_new.graph.nodes():
    centroid = state_new.graph.nodes()[node]['Centroid']
    state_new.graph.nodes()[node]['boundary'] = (centroid[0] in (0, 40) or centroid[1] in (0, 40))
    state_new.graph.nodes()[node]['population'] = 1
    # TODO update for new lattice size

