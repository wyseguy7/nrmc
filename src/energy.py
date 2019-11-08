import os
import random

import mergeSplit

import importlib

importlib.reload(mergeSplit)

def compactnessScore(state, info, changedDistricts = None):
    score = 0
    nodeToDistrict = state["nodeToDistrict"]
    graph = state["graph"]

    districtAdj, borderEdges = mergeSplit.findDistrictAdj(state)

    for dist in state["districts"]:
        if changedDistricts != None and dist in changedDistricts["districts"]:
            district = changedDistricts["districts"][dist]
        else:
            district = state["districts"][dist]
        area = sum([graph.nodes[n]["Area"] for n in district.nodes])
        perim = sum([graph.nodes[n]["BorderLength"] for n in district.nodes
                     if "BorderLength" in graph.nodes[n]])
        for distAdj in districtAdj[dist]:
            perim += sum([graph.edges[e[0], e[1]]["BorderLength"]
                          for e in borderEdges[dist][distAdj]])
        score += perim**2/area
    return score

def cutEnergy(state, info, changedDistricts = None):
    lmbda = info["energy"]["lambda"]
    cutLength = len(state["conflictedEdges"])
    return lmbda**(-cutLength)

def getEnergyFunction(info):
    if "energy" not in info:
        return (lambda state, info: 1)
    energies = []
    if "lambda" in info["energy"]:
        lmbda = info["energy"]["lambda"]
        if lmbda != 0:
            energies.append((1, cutEnergy)) 
    return (lambda state, info, changedDistricts = None: 
                    sum([e[0]*e[1](state, info, changedDistricts) 
                         for e in energies]))

