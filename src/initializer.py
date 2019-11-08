import os
import random

import districtingGraph
import graphGenerator as grphGen

import importlib
importlib.reload(grphGen)


def setIdealPop(state, info):
    mean_huh = info["parameters"]["idealPop"] == "mean" 
    if mean_huh:
        graph = state["graph"]
        popTotal = sum([graph.nodes[n]["Population"] for n in graph.nodes])
        idealPop = popTotal/float(info["parameters"]["districts"])
        state["idealPop"] = idealPop
    else:
        state["idealPop"] = info["parameters"]["idealPop"]
    return state

def setPopBounds(state, info):
    try:
        popDiv = info["constraints"]["populationDeviation"]
        state["maxPop"] = state["idealPop"]*(1 + popDiv)
        state["minPop"] = state["idealPop"]*(1 - popDiv)
    except:
        graph = state["graph"]
        popTotal = sum([graph.nodes[n]["Population"] for n in graph.nodes])
        state["maxPop"] = popTotal
        state["minPop"] = 10.0**-14
    return state

def setPopulation(state, info):
    graph = state["graph"]    
    popTotal = sum([graph.nodes[n]["Population"] for n in graph.nodes])
    state["Population"] = popTotal
    return state

def determineStateInfo(state, info):
    state = setPopulation(state, info)
    state = setIdealPop(state, info)
    state = setPopBounds(state, info)
    return state

def fillMissingInfoFields(info):
    if "step" not in info["parameters"]:
        info["parameters"]["step"] = 0
    if "idealPop" not in info["parameters"]:
        info["parameters"]["idealPop"] = "mean"
    return info

def readCmd(cmd, val, sysargs, type = float, strElse = False):
    
    toReturn = val

    if not strElse:
        if cmd in sysargs:
            ind = sysargs.index(cmd) + 1
            val = type(sysargs[ind])
    else:
        if cmd in sysargs:
            ind = sysargs.index(cmd) + 1
            try:
                val = type(sysargs[ind])
            except:
                val = sysargs[ind]

    return val

def setGraph(pathToData, state):
    geometryDesc = pathToData.split(os.sep)[-1]
    if "BigonDual" in geometryDesc:
        grphGen.generateBigonDualGraphData(pathToData)
    elif "RdDual" in geometryDesc:
        raise Exception("not yet coded")
    elif "SquareLattice" in geometryDesc:
        grphGen.generateSquareLatticeData(pathToData)
    state["graph"] = districtingGraph.set(pathToData)
    return state
    
def setRunParametersFromCommandLine(sysargs = []):

    state = {}

    geometryDesc = "DuplinOnslow"
    numDists = 2
    idealPop = "mean"
    steps = 10**6
    mul = 1
    seed = 912311
    gamma = "_nonTemperedProposal_noFlow_"
    popDivConstraint = 0.1
    compactnessWeight = 0
    lmbda = 1.0/2.63815853031
    lmbda = 1

    mul = readCmd("--mulSeed", mul, sysargs, int)
    gamma = readCmd("--gamma", gamma, sysargs)
    lmbda = readCmd("--lambda", lmbda, sysargs)
    numDists = readCmd("--numDists", numDists, sysargs, int)
    idealPop = readCmd("--idealPop", idealPop, sysargs, strElse = True)
    geometryDesc = readCmd("--geom", geometryDesc, sysargs, str)
    popDivConstraint = readCmd("--popDivConstraint", popDivConstraint, sysargs)
    compactnessWeight = readCmd("--weightPP", compactnessWeight, sysargs)

    seed *= mul
    
    pathToData = os.path.join("..", "data", geometryDesc)
    state = setGraph(pathToData, state)
    
    random.seed(seed)
    rng = random
    
    desc = "gamma" + str(gamma).replace(".", "p") + "_seed" + str(seed) \
          +"_lambda" + str(lmbda)
    if compactnessWeight > 0:
        desc += "_wc" + str(compactnessWeight)
    outDir = os.path.join("..","Output", geometryDesc, desc)
    
    args = {}
    args["rng"] = rng
    args["energy"] = {"lambda" : lmbda}
    args["constraints"] = {"populationDeviation" : popDivConstraint}
    args["parameters"] = {"gamma" : gamma, "idealPop" : idealPop, 
                          "districts" : numDists, "steps" : steps, 
                          "outDir" : outDir}
    # print(args)
    # print(seed)
    return state, args