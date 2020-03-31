import math
import numpy as np
import networkx as nx

import tree
import stateExt

import importlib

importlib.reload(tree)
importlib.reload(stateExt)

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

def getProjections(ce, state):
    nodeToDistrict = state["nodeToDistrict"]
    
    dists = [nodeToDistrict[ce[0]], nodeToDistrict[ce[1]]]
    centroidSumsOG = {d:(state["centroidSums"][d][0],
                         state["centroidSums"][d][1]) for d in dists}
    nodeCountsOG = {d: state["nodeCounts"][d] for d in dists}
    comsOG = {d:(centroidSumsOG[d][0]/nodeCountsOG[d],
                 centroidSumsOG[d][1]/nodeCountsOG[d]) 
              for d in dists}

    #what if we flipped the first node of the edge??
    centroidNode1 = state["graph"].nodes[ce[0]]["Centroid"]
    # print(ce, centroidNode1, dists)
    d1, d2 = dists
    centroidSumsNew = {}
    nodeCountsNew = {d : nodeCountsOG[d] for d in nodeCountsOG}
    centroidSumsNew[d1] = (centroidSumsOG[d1][0] - centroidNode1[0],
                           centroidSumsOG[d1][1] - centroidNode1[1])
    centroidSumsNew[d2] = (centroidSumsOG[d2][0] + centroidNode1[0],
                           centroidSumsOG[d2][1] + centroidNode1[1])
    nodeCountsNew[d1] -= 1
    nodeCountsNew[d2] += 1
    comsNew = {d:(centroidSumsNew[d][0]/nodeCountsNew[d],
                centroidSumsNew[d][1]/nodeCountsNew[d]) 
               for d in dists}

    comsDir = {d:(comsNew[d][0] - comsOG[d][0], comsNew[d][1] - comsOG[d][1])
               for d in dists}
    flowCenter = state["flowCenter"]
    COMAvg = {d:(0.5*(comsNew[d][0] + comsOG[d][0]) - flowCenter[0], 
                 0.5*(comsNew[d][1] + comsOG[d][1]) - flowCenter[1])
                 for d in dists}
    flowVecs = {}
    flowDir = state["flowDir"]
    # print("flowDir", flowDir)
    for d in dists:
        COMdx, COMdy = COMAvg[d]
        l = np.sqrt(COMdx*COMdx + COMdy*COMdy) 

        theta = np.arccos(COMdx/l)
        if COMdy < 0:
            theta = 2*np.pi - theta
        flowVecs[d] = (-np.sin(theta)*flowDir, np.cos(theta)*flowDir)

    projections =[  comsDir[d][0]*flowVecs[d][0] 
                  + comsDir[d][1]*flowVecs[d][1] for d in dists]
    return projections

def directConflictedEdges(state):
    flowedMoves = set([])
    nodeToDistrict = state["nodeToDistrict"]
    for ce in state["conflictedEdges"]:
        # if '1250' not in ce:# or '1005' not in ce:
        #     continue
        projections = getProjections((ce[0], ce[1]), state)
        if sum(projections) > 0 or True:
            flowedMoves.add((ce[0], nodeToDistrict[ce[1]]))
        # elif sum(projections) == 0:
        #     raise Exception("found precise zero")

        projections = getProjections((ce[1], ce[0]), state)
        if sum(projections) > 0 or True:
            flowedMoves.add((ce[1], nodeToDistrict[ce[0]]))
        # elif sum(projections) == 0:
        #     raise Exception("found precise zero")
        # print("flowVec", flowVecs)
        # print("comsog", comsOG)
        # print("comsnew", comsNew)
        # print("comsDir", comsDir)
        # print("projections", projections)
        
    return flowedMoves

def getProjections2(ce, state):
    nodeToDistrict = state["nodeToDistrict"]
    
    dists = [nodeToDistrict[ce[0]], nodeToDistrict[ce[1]]]
    centroidSumsOG = {d:(state["centroidSums"][d][0],
                         state["centroidSums"][d][1]) for d in dists}
    nodeCountsOG = {d: state["nodeCounts"][d] for d in dists}
    comsOG = {d:(centroidSumsOG[d][0]/nodeCountsOG[d],
                 centroidSumsOG[d][1]/nodeCountsOG[d]) 
              for d in dists}

    #what if we flipped the first node of the edge??
    centroidNode1 = state["graph"].nodes[ce[0]]["Centroid"]
    print(ce, centroidNode1, dists)
    d1, d2 = dists
    centroidSumsNew = {}
    nodeCountsNew = {d : nodeCountsOG[d] for d in nodeCountsOG}
    centroidSumsNew[d1] = (centroidSumsOG[d1][0] - centroidNode1[0],
                           centroidSumsOG[d1][1] - centroidNode1[1])
    centroidSumsNew[d2] = (centroidSumsOG[d2][0] + centroidNode1[0],
                           centroidSumsOG[d2][1] + centroidNode1[1])
    nodeCountsNew[d1] -= 1
    nodeCountsNew[d2] += 1
    comsNew = {d:(centroidSumsNew[d][0]/nodeCountsNew[d],
                centroidSumsNew[d][1]/nodeCountsNew[d]) 
               for d in dists}

    comsDir = {d:(comsNew[d][0] - comsOG[d][0], comsNew[d][1] - comsOG[d][1])
               for d in dists}
    flowCenter = state["flowCenter"]
    COMAvg = {d:(0.5*(comsNew[d][0] + comsOG[d][0]) - flowCenter[0], 
                 0.5*(comsNew[d][1] + comsOG[d][1]) - flowCenter[1])
                 for d in dists}
    flowVecs = {}
    flowDir = state["flowDir"]
    print("flowDir", flowDir)
    print("comog", comsOG)
    print("comnew", comsNew)
    print("comdir", comsDir)
    print("comavg", COMAvg)
    for d in dists:
        COMdx, COMdy = COMAvg[d]
        l = np.sqrt(COMdx*COMdx + COMdy*COMdy) 

        theta = np.arccos(COMdx/l)
        if COMdy < 0:
            theta = 2*np.pi - theta
        flowVecs[d] = (-np.sin(theta)*flowDir, np.cos(theta)*flowDir)
    print("flowVecs", flowVecs)
    projections =[  comsDir[d][0]*flowVecs[d][0] 
                  + comsDir[d][1]*flowVecs[d][1] for d in dists]
    print("projections", projections, "sum", sum(projections))
    return projections

def directConflictedEdges2(state):
    flowedMoves = set([])
    nodeToDistrict = state["nodeToDistrict"]
    for ce in state["conflictedEdges"]:
        # if '1250' not in ce:# or '1005' not in ce:
        #     continue
        projections = getProjections2((ce[0], ce[1]), state)
        if sum(projections) > 0:
            flowedMoves.add((ce[0], nodeToDistrict[ce[1]]))
        # elif sum(projections) == 0:
        #     raise Exception("found precise zero")

        projections = getProjections2((ce[1], ce[0]), state)
        if sum(projections) > 0:
            flowedMoves.add((ce[1], nodeToDistrict[ce[0]]))
        # elif sum(projections) == 0:
        #     raise Exception("found precise zero")
        # print("flowVec", flowVecs)
        # print("comsog", comsOG)
        # print("comsnew", comsNew)
        # print("comsDir", comsDir)
        # print("projections", projections)
        
    return flowedMoves

def trimMovesOutsideOfPop(flowedMoves, state):
    minpop = state["minPop"]
    maxpop = state["maxPop"]
    nodeToDistrict = state["nodeToDistrict"]
    trimmedMoves = set([])
    for move in flowedMoves:
        curDist = nodeToDistrict[move[0]]
        newDist = move[1]
        if len(state['districts'][curDist]) - 1 < minpop:
            trimmedMoves.add(move)
            continue 
        if len(state['districts'][newDist]) + 1 > maxpop:
            trimmedMoves.add(move)
    return flowedMoves - trimmedMoves

def trimMovesErasingBrd(flowedMoves, state):
    trimmedMoves = set([])
    nodeToDistrict = state["nodeToDistrict"]
    for move in flowedMoves:
        curDist = nodeToDistrict[move[0]]
        if state["nodesOnBoundary"][curDist] > 1:
            continue
        elif 'BorderLength' in state['graph'].nodes[move[0]]:
            trimmedMoves.add(move)
    return flowedMoves - trimmedMoves

def trimDiscongigousMoves(flowedMoves, state):
    trimmedMoves = set([])
    nodeToDistrict = state["nodeToDistrict"]
    for move in flowedMoves:
        curDist = nodeToDistrict[move[0]]
        # print(state['graph'].neighbors(move[0]))
        # print(state['graph'].nodes[move[0]]["Centroid"])
        nbrsOfCurDist = set([])
        for n in state['graph'].neighbors(move[0]):
            if nodeToDistrict[n] != curDist:
                continue
            nbrsOfCurDist.add(n)
            # print("nghbr", n, state['graph'].nodes[n]["Centroid"])
        nbrsOfCurDist = list(nbrsOfCurDist)
        for ii in range(len(nbrsOfCurDist)):
            for jj in range(ii+1, len(nbrsOfCurDist)):
                ni = nbrsOfCurDist[ii]
                nj = nbrsOfCurDist[jj]
                nbi = set([nb for nb in state['graph'].neighbors(ni) 
                           if nb!=move[0]])
                nbj = set([nb for nb in state['graph'].neighbors(nj) 
                           if nb!=move[0]])
                sharedNbrs = nbi.intersection(nbj)
                if len(sharedNbrs) == 0:
                    continue
                sharedNbrs = set([nb for nb in sharedNbrs 
                                  if nodeToDistrict[nb] == curDist])
                if len(sharedNbrs) == 0:
                    trimmedMoves.add(move)
                    break
    for move in flowedMoves:
        nodeToFlip = move[0]
        curDist = nodeToDistrict[nodeToFlip]
        # print(state['graph'].neighbors(move[0]))
        # print(state['graph'].nodes[move[0]]["Centroid"])
        nbrsOfCurDist = set([])
        for n in state['graph'].neighbors(move[0]):
            if nodeToDistrict[n] != curDist:
                continue
            nbrsOfCurDist.add(n)
            # print("nghbr", n, state['graph'].nodes[n]["Centroid"])
        nbrsOfCurDist = list(nbrsOfCurDist)
        if len(nbrsOfCurDist) == 2:
            centroid1 = state['graph'].nodes[nbrsOfCurDist[0]]["Centroid"]
            centroid2 = state['graph'].nodes[nbrsOfCurDist[1]]["Centroid"]
            if np.abs(centroid1[0]-centroid2[0])==2 or\
               np.abs(centroid1[1]-centroid2[1])==2:
               trimmedMoves.add(move)
    return flowedMoves - trimmedMoves

def proposalProbEdges(flowedMoves, state, lmbda, gamma):
    curLen = len(state["conflictedEdges"])
    flowedMoves = trimMovesOutsideOfPop(flowedMoves, state)
    flowedMoves = trimMovesErasingBrd(flowedMoves, state)
    flowedMoves = trimDiscongigousMoves(flowedMoves, state)

    moveToProb = {}
    sumProb = 0
    # oldConfEdgeCount = len(state["conflictedEdges"])
    for move in flowedMoves:
        delConfEdgeCount = 0
        newDist = move[1]
        for nb in state['graph'].neighbors(move[0]):
            if state["nodeToDistrict"][nb] == newDist:
                delConfEdgeCount -= 1
            else:
                delConfEdgeCount += 1
        # prob = lmbda**(gamma*delConfEdgeCount) 
        prob = np.exp(-0.5*lmbda*delConfEdgeCount) 
        # if 'BorderLength' in state['graph'].nodes[move[0]]:
        #     prob  *= 2
                                         # if num conf edge decreases with move, 
                                         # want probability to increase. 
                                         # since lmbda < 1, lmbda^{neg} > 1
                                         # and lmbda^{pos} < 1
        moveToProb[move] = prob
        sumProb += prob
        # print(delConfEdgeCount, prob, move, 
        #       state['graph'].nodes[move[0]]["Centroid"])
    moveToProb = {mv : moveToProb[mv]/sumProb for mv in moveToProb}
    return moveToProb

def proposeMove(temperedMoveList, info):
    rng = info["rng"]
    uq = rng.random()
    cumSum = 0

    # print("len tempered moves", len(temperedMoveList))
    keys = list(temperedMoveList.keys())
    keys.sort()
    for move in keys:
        prob = temperedMoveList[move]
        cumSum += prob
        # print(prob, cumSum, uq, move)
        if cumSum > uq:
            # print("returning")
            return move, prob
    toRet = [list(temperedMoveList.keys())[-1], 
             list(temperedMoveList.values())[-1]]
    return toRet

def calcRevProposalProb(state, move, info):
    nodeToFlip = move[0]
    curDist = state["nodeToDistrict"][move[0]]
    newDist = move[1]
    reverseMove = (nodeToFlip, curDist)

    # if nodeToFlip == "840":
    #     flowedMoves = directConflictedEdges2(state)
    #     print(flowedMoves)
    #     print()

    # print("og conf edges", state["conflictedEdges"])
    state["flowDir"] *= -1
    state = stateExt.updateState(move, state, info)

    # if nodeToFlip == "840":
    #     flowedMoves = directConflictedEdges2(state)
    #     print(flowedMoves)
    #     print()
    #     exit()

    # print("new conf edges", state["conflictedEdges"])
    flowedMoves = directConflictedEdges(state)
    # print()
    # print("flowedmoves", flowedMoves)
    # exit()
    lmbda, gamma = info["energy"]["lambda"], info["parameters"]["gamma"]
    temperedMoveList = proposalProbEdges(flowedMoves, state, lmbda, gamma)
    
    state["flowDir"] *= -1
    stateExt.updateState(reverseMove, state, info)

    revProb = temperedMoveList[reverseMove]
    return revProb

def calcDelProb(state, move, lmbda):
    delConfEdgeCount = 0
    newDist = move[1]
    for nb in state['graph'].neighbors(move[0]):
        if state["nodeToDistrict"][nb] == newDist:
            delConfEdgeCount -= 1
        else:
            delConfEdgeCount += 1
    # prob = lmbda**(delConfEdgeCount)
    # print("delCE", delConfEdgeCount)
    prob = np.exp(-lmbda*delConfEdgeCount) #np.exp(-lmbda*(new-old))
    return prob

def COM_Flow(lmbda, gamma):
    def f(state, info):
        '''Returns a single merge-split step with associated probability,
           including spanning trees.'''
        flowedMoves = directConflictedEdges(state)
        temperedMoveList = proposalProbEdges(flowedMoves, state, lmbda, gamma)

        # if state["step"] == 1408:
        #     flowedMoves = directConflictedEdges2(state)
        #     print("flowedMoves", flowedMoves)
        #     print("conflictedEdges", state["conflictedEdges"])
        #     print("len conflictedEdges", len(state["conflictedEdges"]))
        #     print("temperedMoveList", temperedMoveList)
        #     exit()

        # print(temperedMoveList)
        move, forwardProposalProb = proposeMove(temperedMoveList, info)
        # print(move)
        # exit()
        reverseProposalProb = calcRevProposalProb(state, move, info) 
        delProbBackwardOverForward = calcDelProb(state, move, lmbda)
        p = reverseProposalProb/forwardProposalProb
        p *= delProbBackwardOverForward
        # print(reverseProposalProb, forwardProposalProb, delProbBackwardOverForward)
        # print(p)
        # print(move)
        # exit()
        return (move, p)
    return f

def define(mcmcArgs):
    lmbda = mcmcArgs["energy"]["lambda"]
    gamma = mcmcArgs["parameters"]["gamma"]
    mcmcArgs["proposal"] = "centerOfMassFlow"
    return COM_Flow(lmbda, gamma), mcmcArgs
