
def checkPopulation(state):
    graph = state["graph"]

    for di in state["districts"]:
        district = state["districts"][di]
        distPop = sum([graph.nodes[n]["Population"] for n in district.nodes])
        if distPop < state["minPop"] or distPop > state["maxPop"]:
            return False
    return True


def simply_connected(state, node_id, old_color, new_color):
    # asssuming that a proposal will result in a graph being connected, check that it will be simply connected
    # this occurs iff there is another node district:old_color that touches either boundary or another color
    # update_contested_edges(state) # do we need this here or can we pull it somewhere else?

    if state.boundary_node_counter[old_color] > 1:
        return True # this handles the vast majority of cases

    smaller = state.color_to_node[old_color] - {node_id}
    contested_nodes = ({i[0] for i in state.contested_edges}.union({i[1] for i in state.contested_edges})).intersection(
        smaller) # nodes that are contested and are in the shrunk district
    for node in contested_nodes:
        # is this node either on the boundary, or does it touch some other color than
        if state.graph.nodes()[node]['boundary'] or len([i for i in state.graph.neighbors(node)
                                                         if state.node_to_color[i] not in (old_color, new_color)]) > 0:

            # about to return True - should be able to check that

            return True
    return False