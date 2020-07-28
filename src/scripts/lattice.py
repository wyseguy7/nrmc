import sys
import pickle

import numpy as np
import networkx as nx

from src.state import greedy_graph_coloring


# sys.path.append('C:\\Users\\wyseg\\nonreversiblecodebase\\src\\')
sys.path.append('/home/grad/etw16/nonreversiblecodebase/') # TODO make this less garbage-y
sys.path.append('/home/grad/etw16/nonreversiblecodebase/src/legacy/') # TODO make this less garbage-y
sys.path.append('/home/grad/etw16/nonreversiblecodebase/src/') # TODO make this less garbage-y

from src.state import State


def create_square_lattice(n=40, boundary=None, num_districts=2, **kwargs):
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

        district_angles = np.linspace(-np.pi, np.pi, num=num_districts+1)
        center = (n-1)/2 # the centroid - since the node centroids are from 0,0 to n-1, n-1

        # flipping the order ensures a horizontal cut, for backward compatibility
        angles = {node_id: np.arctan2(g.nodes()[node_id]['Centroid'][1]-center,
                                      g.nodes()[node_id]['Centroid'][0]-center) for node_id in g.nodes()}

        # subtract 1 here to ensure zero-indexing of districts
        node_to_color = {node_id: np.digitize(angle, district_angles)-1 for node_id, angle in angles.items()}

    else:
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


