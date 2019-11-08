
def checkPopulation(state):
    graph = state["graph"]

    for di in state["districts"]:
        district = state["districts"][di]
        distPop = sum([graph.nodes[n]["Population"] for n in district.nodes])
        if distPop < state["minPop"] or distPop > state["maxPop"]:
            return False
    return True