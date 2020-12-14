import sys

import numpy as np
import networkx as nx

# sys.path.append('C:\\Users\\wyseg\\nonreversiblecodebase\\src\\')
sys.path.append('/home/grad/etw16/nonreversiblecodebase/') # TODO make this less garbage-y
sys.path.append('/home/grad/etw16/nonreversiblecodebase/src/legacy/') # TODO make this less garbage-y
sys.path.append('/home/grad/etw16/nonreversiblecodebase/src/') # TODO make this less garbage-y

from .state import State


def create_square_lattice(n=40, boundary=None, num_districts=2, **kwargs):
    g = nx.Graph()
    for i in range(n):
        for j in range(n):
            node_id = i * n + j
            on_boundary = (i in (0, n-1) or j in (0, n-1))
            g.add_node(node_id, Centroid=np.array([i, j], dtype='d'), boundary=on_boundary, population=1, area=1,
                       external_border = int(i in (0, n-1)) + int(j in (0, n-1)))

            if i != 0:
                # add western node
                g.add_edge(node_id, (i - 1) * n + j, border_length=1)

            if j != 0:
                g.add_edge(node_id, i * n + j - 1, border_length=1)
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


    return State(g, node_to_color, graph_type='lattice', **kwargs)

def polar_to_cartesian(r, theta):
    # requires theta in (-pi, pi)

    return r*np.sin(theta), r*np.cos(theta)


def create_star(n=10, num_points = 4, radius_adjustment=0.1, num_districts=2, **kwargs):

    g = nx.Graph()

    # build in a clockwise fashion
    node_idx_counter = 0
    theta = -np.pi # start at the back
    generic_args = {'population': 1, 'area': 1, 'boundary': True, 'external_border': 0}

    # create initial values
    point_lookup = {}

    for i in range(num_points):


        g.add_node(node_idx_counter, Centroid=np.array(polar_to_cartesian(1, theta), dtype='d'), **generic_args)
        point_lookup[theta] = node_idx_counter # record the lookup
        theta += 2* np.pi/num_points # rotate to set up
        node_idx_counter += 1

    point_lookup[np.pi] = point_lookup[-np.pi]
    theta_loc = -np.pi + np.pi/num_points

    for i in range(num_points):

        last_idx = point_lookup[theta_loc - np.pi/num_points]
        next_idx = point_lookup[theta_loc + np.pi/num_points]

        for j in range(n):

            g.add_node(node_idx_counter, Centroid=np.array(polar_to_cartesian(1+radius_adjustment*j, theta_loc), dtype='d'), **generic_args)
            g.add_edge(node_idx_counter, last_idx)
            g.add_edge(node_idx_counter, next_idx)
            node_idx_counter += 1

        theta_loc += 2*np.pi/num_points # hop along!


    district_angles = np.linspace(-np.pi, np.pi+1e-10, num=num_districts+1) # adding 1e-10 to deal with weird rounding issue

    center = np.mean([g.nodes()[node_id]['Centroid'] for node_id in g.nodes()]) # hope this works properly

    # flipping the order ensures a horizontal cut, for backward compatibility
    angles = {node_id: np.arctan2(g.nodes()[node_id]['Centroid'][1] - center,
                                  g.nodes()[node_id]['Centroid'][0] - center) for node_id in g.nodes()}

    # subtract 1 here to ensure zero-indexing of districts
    node_to_color = {node_id: np.digitize(angle, district_angles) - 1 for node_id, angle in angles.items()}

    return State(g, node_to_color, graph_type='point', **kwargs)