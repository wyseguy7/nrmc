import tree

import importlib

importlib.reload(tree)

def setExtensionData(state, info):
    state["extensions"] = set([])
    if info["proposal"] == "mergeSplit":
        state["extensions"].add("spanningTrees")
        if info["parameters"]["gamma"] != 0:
            state["extensions"].add("spanningTreeCounts")
    if info["proposal"] == "centerOfMassFlow":
        state["extensions"].add("flowDir")

    return state

def extendState(state, info, computeE):
    state = setExtensionData(state, info)

    rng = info["rng"]

    if "spanningTrees" in state["extensions"]:
        state["districtTrees"] = {}
        for di in state["districts"]:
            district = state["districts"][di]
            state["districtTrees"][di] = tree.wilson(district, rng)
    if "spanningTreeCounts" in state["extensions"]:
        state["spanningTreeCounts"] = {}
        for di in state["districts"]:
            district = state["districts"][di]
            state["spanningTreeCounts"][di] = tree.nspanning(district)

    state["energy"] = computeE(state, info)

    return state

def updateState(move, state, info):

    nodeToFlip = move[0]
    curDist = state['nodeToDistrict'][nodeToFlip]
    newDist = move[1]
    nodeCentroid = state['graph'].nodes[nodeToFlip]["Centroid"]

    # print("flipping", nodeToFlip, curDist, newDist, nodeCentroid)

    state['nodeToDistrict'][nodeToFlip] = newDist
    state['districts'][curDist].remove_node(nodeToFlip)
    state['districts'][newDist].add_node(nodeToFlip)

    # print("conf edges", state["conflictedEdges"])
    for nb in state['graph'].neighbors(nodeToFlip):
        if state['nodeToDistrict'][nb] == newDist:
            # print("removing", nb, nodeToFlip)
            try:
                state["conflictedEdges"].remove((nb, nodeToFlip))
            except:
                state["conflictedEdges"].remove((nodeToFlip, nb))
        else:
            # print("adding", nb, nodeToFlip)
            state["conflictedEdges"].add((nodeToFlip, nb))

    
    if 'BorderLength' in state['graph'].nodes[nodeToFlip]:
        state["nodesOnBoundary"][curDist] -= 1
        state["nodesOnBoundary"][newDist] += 1
    
    state["centroidSums"][curDist] = \
                           (state["centroidSums"][curDist][0] - nodeCentroid[0],
                            state["centroidSums"][curDist][1] - nodeCentroid[1])
    state["centroidSums"][newDist] = \
                           (state["centroidSums"][newDist][0] + nodeCentroid[0],
                            state["centroidSums"][newDist][1] + nodeCentroid[1])
    state["nodeCounts"][curDist] -= 1
    state["nodeCounts"][newDist] += 1
    return state
