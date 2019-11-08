import math
import networkx as nx

import tree

import importlib

importlib.reload(tree)

exp = lambda x: math.exp(min(x, 700)) #>~700 overflows

################################################################################

def findDistrictAdj(state):
    '''Generates a district adjacency graph formatted as a list of sets
       and the border vertices formatted as a list of sets.'''
    assignment = state["nodeToDistrict"]
    graph = state["graph"]
    districtAdj = {}
    borderEdges = {}
    districts = state["districts"]
    for di in districts:
        districtAdj[di] = set()
        borderEdges[di] = {dj : set() for dj in districts if dj != di}
        for node in districts[di].nodes:
            for nbrNode in graph.neighbors(node):
                dj = assignment[nbrNode]
                if dj != di:
                    districtAdj[di].add(assignment[nbrNode])
                    borderEdges[di][dj].add((node, nbrNode))
    return districtAdj, borderEdges

def proposeAdjDists(districtAdj, info):
    numDists = info["parameters"]["districts"]
    rng = info["rng"]
    
    d1 = rng.randint(0, numDists-1)
    d2 = rng.choice(sorted(list(districtAdj[d1])))
    return d1, d2

def getMergedTree(state, info, d1, d2):
    districts, graph = state["districts"], state["graph"]
    mergedNodes = set(districts[d1]).union(set(districts[d2]))
    mergedGraph = graph.subgraph(mergedNodes)
    mergedTree = tree.wilson(mergedGraph, info["rng"])
    mergedTreePop = sum([graph.nodes[n]["Population"] for n in mergedTree])
    return mergedTree, mergedTreePop

def mergeSplit(state, info):
    '''Returns a single merge-split step with associated probability,
       not including spanning trees.'''
    print("begin merge-split")
    newdistricts = {}
    districtAdj, borderEdges = findDistrictAdj(state)
    d1, d2 = proposeAdjDists(districtAdj, info)
    
    mergedTree, mergedTreePop = getMergedTree(state, info, d1, d2)
    cutEdges, edgeWeights = tree.edgeCuts(mergedTree, mergedTreePop, 
                                          state, info)
    if not cutEdges:
        return (None, 0, d1, d2)
    rng = info["rng"]
    orderedCutEdges = sorted(cutEdges, key=lambda e:''.join(sorted(list(e))))
    cutEdge = rng.choice(orderedCutEdges)
    mergedTree.remove_edge(*cutEdge)

    graph = state["graph"]
    oldD = [state["districts"][d1], state["districts"][d2]]
    newD = [mergedTree.subgraph(n) for n in nx.connected_components(mergedTree)]
    oldAssignment = state["nodeToDistrict"]
    newAssignment = {n : oldAssignment[n] for n in oldAssignment}
    for di, nD in zip([d1,d2], newD):
        for n in nD.nodes:
            newAssignment[n] = di
    state["nodeToDistrict"] = newAssignment

    districts = state["districts"]
    districts[d1] = newD[0]
    districts[d2] = newD[1]
    newdistrictAdj, newBorderEdges = findDistrictAdj(state)
    districts[d1] = oldD[0]
    districts[d2] = oldD[1]
    state["nodeToDistrict"] = oldAssignment
    print("middle mergesplit")

    p =  (1.0/len(newdistrictAdj[d1]) + 1.0/len(newdistrictAdj[d2])) \
        /(1.0/len(districtAdj[d1]) + 1.0/len(districtAdj[d2])) 
    brdrEdges_ij = borderEdges[d1][d2]
    newbrdrEdges_ij = newBorderEdges[d1][d2]

    backwardPcut = tree.crossEdgeProbSum(state["districtTrees"][d1], 
                                         state["districtTrees"][d2], 
                                         mergedTreePop, brdrEdges_ij, state, 
                                         info)
    forwardPcut = tree.crossEdgeProbSum(newD[0], newD[1], mergedTreePop, 
                                        newbrdrEdges_ij, state, info)
    p *= backwardPcut/forwardPcut
    newdistricts["districtTrees"] = {d1 : newD[0], d2 : newD[1]}
    newdistricts["districts"] = {d1 : graph.subgraph(newD[0].nodes), 
                                 d2 : graph.subgraph(newD[1].nodes)}
    newdistricts["nodeToDistrict"] = newAssignment
    print("end merge-split")

    return (newdistricts, p, d1, d2)


def mergeSplitGamma(gamma):
    def f(state, info):
        '''Returns a single merge-split step with associated probability,
           including spanning trees.'''
        (newdistricts, p, d1, d2) = mergeSplit(state, info)
        if p:
            graph = state["graph"]
            nD1 = newdistricts["districts"][d1]
            nD2 = newdistricts["districts"][d2]
            spTrCnt1 = tree.nspanning(nD1)
            spTrCnt2 = tree.nspanning(nD2)
            newdistricts["spanningTreeCounts"] = {d1 : spTrCnt1, d2 : spTrCnt2}
            spTrCnt1Old = state["spanningTreeCounts"][d1]
            spTrCnt2Old = state["spanningTreeCounts"][d2]
            p *= exp(gamma*(spTrCnt1Old + spTrCnt2Old - spTrCnt1 - spTrCnt2))
        return (newdistricts, p, d1, d2)
    return f

def define(mcmcArgs):
    gamma = mcmcArgs["parameters"]["gamma"]
    mcmcArgs["proposal"] = "mergeSplit"
    if gamma == 0:
        return mergeSplit, mcmcArgs
    else:
        return mergeSplitGamma(gamma), mcmcArgs
