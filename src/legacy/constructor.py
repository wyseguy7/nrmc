import networkx as nx
import operator

# from src import constraints, initializer, tree
import constraints
import initializer
import tree

import importlib
importlib.reload(constraints) # why is this here?
importlib.reload(initializer)
importlib.reload(tree)


def searchForDistrict(graph, rng, state, info, remainingPop, attemptPerDist, 
                      boolop):    
    for attempt in range(attemptPerDist):
        graphTree = tree.wilson(graph, rng)

        cutSet, edgeWeights = {}, {}
        try:
            cutSet, edgeWeights = tree.edgeCuts(graphTree, remainingPop, state, 
                                                info, boolop)
        except:
            state = initializer.determineStateInfo(state, info)
            cutSet, edgeWeights = tree.edgeCuts(graphTree, remainingPop, graph, 
                                                info, boolop)
        if cutSet:
            break
    return cutSet, edgeWeights, graphTree

def extractDistrictsFromCutTree(graphTree, graph, state, districts):
    distGraph = state["graph"]

    for conComp in nx.connected_components(graphTree):
        popConComp = sum([distGraph.nodes[n]["Population"] 
                          for n in conComp])
        if state["minPop"] <= popConComp <= state["maxPop"]:
            districts[len(districts)] = graph.subgraph(conComp).copy()
            graph.remove_nodes_from(conComp)
    return districts, graph

def assignDistricts(districts, state):
    state["districts"] = districts

    nodeToDistrict = {}
    for dist in districts:
        for n in districts[dist].nodes:
            nodeToDistrict[n] = dist
    state["nodeToDistrict"] = nodeToDistrict

    return state

def attemptMakePlan(state, info, attemptPerDist = 20):
    if "seedAttemptsByDist" in info["parameters"]:
        attemptPerDist = info["parameters"]["seedAttemptsByDist"]
    numDist = info["parameters"]["districts"]
    rng = info["rng"]
    distGraph = state["graph"]
    graph = distGraph.copy()

    districts = {}
    
    cuts = numDist - 1
    for distCut in range(cuts):
        boolop = operator.and_ if distCut == cuts-1 else operator.or_

        remainingPop = sum([distGraph.nodes[n]["Population"] 
                            for n in graph.nodes])
        
        cutSet, edgeWeights, graphTree = searchForDistrict(graph, rng, state, 
                                                           info, remainingPop,
                                                           attemptPerDist, 
                                                           boolop)
        if not cutSet:
            return state, False
    
        orderedCutSet = sorted(list(cutSet), 
                               key=lambda e:''.join(sorted(list(e))))
        e = rng.choice(orderedCutSet)
        
        graphTree.remove_edge(*e)

        districts, graph = extractDistrictsFromCutTree(graphTree, graph, state, 
                                                       districts)

    state = assignDistricts(districts, state)
    
    popCheck = constraints.checkPopulation(state)
    distCheck = len(districts) == numDist
    
    return state, popCheck and distCheck

def contructPlan(state, info, maxAttempts = 1000):
    '''Finds a random districting.'''

    for attempt in range(maxAttempts):
        state, success = attemptMakePlan(state, info)
        if success:
            break
    if not success:
        raise Exception("Could not find acceptable initial state")
    return state

def splitSquareLattice(state, info):
    graph = state["graph"]
    minx = min([graph.nodes[ni]["Centroid"][0] for ni in graph.nodes]) # TODO could find min, max simultaneously with one walk
    maxx = max([graph.nodes[ni]["Centroid"][0] for ni in graph.nodes])
    miny = min([graph.nodes[ni]["Centroid"][1] for ni in graph.nodes])
    maxy = max([graph.nodes[ni]["Centroid"][1] for ni in graph.nodes])

    halfwayx = (minx + maxx)*0.5
    halfwayy = (miny + maxy)*0.5
    districts = {}
    districts[0] = graph.subgraph([ni for ni in graph.nodes 
                       if graph.nodes[ni]["Centroid"][1] < halfwayy or 
                         (graph.nodes[ni]["Centroid"][1] == halfwayy and 
                          graph.nodes[ni]["Centroid"][0] >= halfwayx)]).copy()
    districts[1] = graph.subgraph([ni for ni in graph.nodes 
                       if graph.nodes[ni]["Centroid"][1] > halfwayy or
                         (graph.nodes[ni]["Centroid"][1] == halfwayy and 
                          graph.nodes[ni]["Centroid"][0] < halfwayx)]).copy()
    state = assignDistricts(districts, state)

    # for ni in list(districts[0].nodes):
    #     print("dist 0", ni, graph.nodes[ni]["Centroid"])
    # for ni in list(districts[1].nodes):
    #     print("dist 1", ni, graph.nodes[ni]["Centroid"])
    # exit()
    d1NodeSet = set(list(districts[0].nodes))
    d2NodeSet = set(list(districts[1].nodes))
    if len(d1NodeSet.intersection(d2NodeSet)) != 0:
        raise Exception("Faulty partition")

    centroidSums = {}
    nodeCounts = {}
    com0x = sum([graph.nodes[ni]["Centroid"][0] for ni in districts[0].nodes])
    com0y = sum([graph.nodes[ni]["Centroid"][1] for ni in districts[0].nodes])
    nodes0 = len(districts[0].nodes)
    centroidSums[0] = (com0x, com0y)
    nodeCounts[0] = nodes0

    com1x = sum([graph.nodes[ni]["Centroid"][0] for ni in districts[1].nodes])
    com1y = sum([graph.nodes[ni]["Centroid"][1] for ni in districts[1].nodes])
    nodes1 = len(districts[1].nodes)
    centroidSums[1] = (com1x, com1y)
    nodeCounts[1] = nodes1

    state["centroidSums"] = centroidSums
    state["nodeCounts"] = nodeCounts

    nodesOnBoundary = {}
    conflictedEdges = {}

    N_x = int(maxx - minx + 0.1)
    N_y = int(maxy - miny + 0.1)
    for ix in range(N_x + 1):
        ni0 = str(ix +  (N_x+1)*0)
        ni1 = str(ix +  (N_x+1)*(N_y))
        state["graph"].nodes[ni0]['BorderLength'] = 1
        state["graph"].nodes[ni1]['BorderLength'] = 1
    for iy in range(N_y + 1):
        ni0 = str(0 +  (N_x+1)*iy)
        ni1 = str(N_x +(N_x+1)*iy)
        state["graph"].nodes[ni0]['BorderLength'] = 1
        state["graph"].nodes[ni1]['BorderLength'] = 1

    nodesOnBoundary[0] = sum([int('BorderLength' in graph.nodes[ni]) 
                              for ni in districts[0].nodes])
    nodesOnBoundary[1] = sum([int('BorderLength' in graph.nodes[ni]) 
                              for ni in districts[1].nodes])
    state["nodesOnBoundary"] = nodesOnBoundary
    nodeToDistrict = state["nodeToDistrict"]
    conflictedEdges = set([ei for ei in graph.edges 
                           if nodeToDistrict[ei[0]] != nodeToDistrict[ei[1]]])
    state["conflictedEdges"] = conflictedEdges
    state["nodesOnBoundary"] = nodesOnBoundary

    state["flowDir"] = 1
    state["flowCenter"] = ((maxx + minx)*0.5 + 0.001213, 
                           (maxy + miny)*0.5 + 0.0081209213) #hacky shifts
    state["flipCountsPerNode"] = {ni : 0 for ni in list(graph.nodes)}

    return state

