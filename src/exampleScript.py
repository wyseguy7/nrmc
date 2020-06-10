import os
import random
import sys

import constructor
import districtingGraph
import initializer
import centerOfMassFlow
import metropolisHastings
# import dataWriter

from importlib import reload

reload(constructor)
reload(districtingGraph)
reload(initializer)
reload(centerOfMassFlow)
reload(metropolisHastings)

print('initializing run...')
state, args = initializer.setRunParametersFromCommandLine(sys.argv)
state["step"] = 0
proposal, args = centerOfMassFlow.define(args)
info = args

info = initializer.fillMissingInfoFields(info)
state = initializer.determineStateInfo(state, info)
# state = constructor.contructPlan(state, info)
state = constructor.splitSquareLattice(state, info) ## terrible code; trying to move fast
# state = constructor.singleCornerDistSquareLattice(state, info) ## terrible code; trying to move fast
print(state.keys())
print(args)
print(info)
print('run initialized...')

print('starting chain...')
print(info["parameters"]["step"], info["parameters"]["steps"])
state = metropolisHastings.run(state, proposal, info)
print('finishing chain...')
