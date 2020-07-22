import sys
import pickle

import numpy as np
import networkx as nx

# sys.path.append('C:\\Users\\wyseg\\nonreversiblecodebase\\src\\')
sys.path.append('/home/grad/etw16/nonreversiblecodebase/') # TODO make this less garbage-y
sys.path.append('/home/grad/etw16/nonreversiblecodebase/src/legacy/') # TODO make this less garbage-y
sys.path.append('/home/grad/etw16/nonreversiblecodebase/src/') # TODO make this less garbage-y

from src.state import State


def create_square_lattice(n=40, boundary=None, **kwargs):
    g = nx.Graph()
    for i in range(n):
        for j in range(n):
            node_id = i * n + j
            on_boundary = (i in (0, n-1) or j in (0, n-1))
            g.add_node(node_id, Centroid=np.array([i, j], dtype='d'), boundary=on_boundary, population=1)

            if i != 0:
                # add western node
                g.add_edge(node_id, (i - 1) * n + j)

            if j != 0:
                g.add_edge(node_id, i * n + j - 1)
                # add southern node

    if boundary is None:
        # construct the classic cut boundary
        lb = int(n / 2)
        ub = int(n / 2) + 1
        boundary = [(u, v) for u, v in g.edges if
                    g.nodes[u]['Centroid'][1] in (lb, ub) and g.nodes[v]['Centroid'][1] in (lb, ub)]

    removed_edges = {}
    for u, v in boundary:  # list of conflicted edges

        removed_edges[(u, v)] = g.edges[(u, v)]  # store relevant data
        g.remove_edge(u, v)

    # nodes_to_color = g.nodes()
    color_to_node = dict()
    node_to_color = dict()

    for district_id, comp in enumerate(nx.connected_components(g)):

        color_to_node[district_id] = comp

        for node in comp:
            node_to_color[node] = district_id

    # add the edges back in
    for (u, v), data in removed_edges.items():
        g.add_edge(u, v, **data)


    return State(g, node_to_color, **kwargs)


