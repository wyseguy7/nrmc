import collections
import os
import itertools
import random
import json
from collections.abc import Iterable
from types import GeneratorType

import networkx as nx
import numpy as np
import pandas as pd
from networkx.readwrite.json_graph import node_link_graph, node_link_data

CENTROID_DIM_LENGTH = 2 # TODO do we need a settings.py file?
try:
    from .biconnected import calculate_com_inner
    cython_biconnected = True
except ImportError:
    cython_biconnected = False

def np_to_native(o):
    if isinstance(o, np.integer):
        return int(o)
    elif isinstance(o, np.floating):
        return float(o)
    elif isinstance(o, np.ndarray):
        return o.tolist()
    elif isinstance(o, dict):
        return {np_to_native_keys(k):np_to_native(v) for k, v in o.items()}
    elif isinstance(o, str):
        return o # this is fine, but don't recurse through iterables
    elif isinstance(o, (Iterable, GeneratorType)):
        return tuple([np_to_native(v) for v in o]) #
    else:
        return o


def np_to_native_keys(o):
    if isinstance(o, (int, np.integer)):
        return int(o)
    elif isinstance(o, (float, np.floating)):
        return float(o)
    else:
        return str(o) # seems like best option


class State(object):



    def __init__(self, graph, coloring, log_contested_edges = True, include_external_border = True,
                 coerce_int = True, apd=0.1, ideal_pop = None, involution = 1, graph_type='lattice', **kwargs):


        if coerce_int:
            # this is important to ensure it's compatible with our Cython speedups
            relabeling = {i: int(i) for i in graph.nodes()}
            coloring = {int(k):v for k,v in coloring.items()}
            graph = nx.relabel_nodes(graph, relabeling, copy=True)

        self.coerce_int = coerce_int
        self.graph = graph
        self.graph_type = graph_type

        self.node_to_color = coloring
        d = collections.defaultdict(set)
        for node_id, district_id in coloring.items():
            d[district_id].add(node_id)
        self.color_to_node = dict(d)

        num_districts = len(self.color_to_node)

        population_total = sum(graph.nodes()[node]['population'] for node in graph.nodes)
        minimum_population = population_total/num_districts - population_total*apd/2
        maximum_population = population_total/num_districts + population_total*apd/2

        self.minimum_population = minimum_population
        self.maximum_population = maximum_population
        self.iteration = 0
        self.state_log = []
        self.move_log = [] # move log is state_log plus 'None' each time we make an involution
        self.ideal_pop = ideal_pop
        self.involution = involution
        self.include_external_border = include_external_border

        self.log_contested_edges = log_contested_edges # rip these out into a mixin?
        self.contested_edge_counter = collections.defaultdict(int)
        self.contested_node_counter = collections.defaultdict(int)


        self.connected_true_counter = [] # TODO remove after finished debugging
        self.connected_false_counter = []

        for name, kwarg in kwargs.items():
            setattr(self, name, kwarg) # set attribute as required

    @property
    def _json_dict(self):

        # these just won't get serialized as they are difficult to initialize or store properly, mostly the updated attrs
        ignore = {"district_boundary", "district_boundary_updated",  "com_centroid", "com_updated", "contested_edges",
                  "contested_nodes", "contested_edges_updated", "boundary_node_counter", "boundary_node_updated",
                  "articulation_points", "articulation_points_updated", "adj_mapping", "adj_mapping_full", "perimeter_computer"
        }

        custom_dict = {'graph': node_link_data(self.graph),
                       'color_to_node': {k: list(v) for k, v in self.color_to_node.items()}, # can't have sets
                       }
        other_dict = {key: value for key, value in self.__dict__.items() if key not in custom_dict and key not in ignore}
        other_dict.update(custom_dict)
        return np_to_native(other_dict)
        # return json.dumps(other_dict)

    @classmethod
    def from_json(cls, js):
        # js is the nested, already parsed dictionary object
        graph = node_link_graph(js['graph'])
        return cls(graph, js['node_to_color'], **{k:v for k,v in js.items() if k != 'graph'})


    @classmethod
    def from_folder(cls, folder_path, num_districts=3, apd=0.1, **kwargs):


        files = os.listdir(folder_path)

        # TODO make these into a function, this is horrible
        border_file = [i for i in files if 'BORDERLENGTHS' in i][0]
        borders = pd.read_csv(os.path.join(folder_path, border_file), sep='\\t', header=None, names=['node_id', 'other_node','border_length'])
        centroids_file = [i for i in files if 'CENTROIDS' in i][0]
        centroids = pd.read_csv(os.path.join(folder_path, centroids_file), sep='\\t', header=None, names=['node_id', 'x','y'])
        centroids_dict = {node_id: {'x': x, 'y': y} for node_id, x, y in centroids.itertuples(index=False)}
        population_file = [i for i in files if 'POPULATION' in i][0]
        population = pd.read_csv(os.path.join(folder_path, population_file), sep='\\t', header=None, names=['node_id', 'population'])
        pop_dict = {node_id: {'population': population} for node_id, population in population.itertuples(index=False)}
        area_file = [i for i in files if 'AREAS' in i][0]
        area = pd.read_csv(os.path.join(folder_path, area_file), sep='\\t', header=None, names=['node_id', 'area'])
        area_dict = {node_id: {'area': area} for node_id, area in area.itertuples(index=False)}

        graph = nx.Graph()

        for node_id, other_node, border_length in borders.itertuples(index=False):
            if node_id != -1:
                graph.add_edge(node_id, other_node, border_length=border_length)
            else:
                graph.add_node(other_node, boundary=True, external_border=border_length)
                # need to make sure we get boundary correct here


        # guarantee we have a boundary entry for each
        for node_id in graph.nodes():

            graph.nodes()[node_id]['Centroid'] = np.array([centroids_dict[node_id]['x'],
                                                           centroids_dict[node_id]['y']] , dtype='d')

            graph.nodes()[node_id]['population'] = pop_dict[node_id]['population']
            graph.nodes()[node_id]['area'] = area_dict[node_id]['area']


            if 'boundary' not in graph.nodes()[node_id]:
                graph.nodes()[node_id]['boundary'] = False
                graph.nodes()[node_id]['external_border'] = 0 # TODO we can eliminate one of these to simplify

        # print(len(graph.nodes))

        population_total = sum(graph.nodes()[node]['population'] for node in graph.nodes)

        minimum_population = population_total/num_districts - population_total*apd/2
        maximum_population = population_total/num_districts + population_total*apd/2

        counter = 0
        loop_max = 100
        # loop until we achieve a valid coloring
        while True:
            counter += 1
            valid_coloring = True
            coloring = greedy_graph_coloring(graph, num_districts=num_districts)
            # print(coloring)
            # check minimum population is valid
            for color, nodes in coloring.items():
                if sum(graph.nodes()[node_id]['population'] for node_id in nodes) < minimum_population or (
                    sum(graph.nodes()[node_id]['population'] for node_id in nodes) > maximum_population):
                    valid_coloring = False
                    break

            if counter > loop_max:
                raise ValueError("Exceeded maximum number of allowable attempts")

            if valid_coloring:
                break

        # produce reverse lookup
        node_to_color = {}
        for district_id, nodes in coloring.items():
            node_to_color.update({node_id: district_id for node_id in nodes})

        return State(graph, coloring=node_to_color, apd=apd, **kwargs)



    def check_connected_lookup(self, proposal):
        # TODO finish this off so we can use it

        node_id, old_color, new_color = proposal

        # check that smaller one is still connected
        district_nodes = self.color_to_node[old_color].difference(
            {node_id})  # all of the old_color nodes without the one we're removing
        to_connect = district_nodes.intersection(
            set(self.graph.neighbors(node_id)))  # need to show each of these are still connected

        if not to_connect:  # TODO should this check get run here or should it be guaranteed?
            return False  # we deleted the last node of a district, this is illegal

        init = min(to_connect)
        to_connect = {i for i in to_connect if i!=init or (init, i) in self.start_stop_to_chain}

        to_search = [init]  # ordered list of nodes available to search
        visited_set = {init}

        connected_counter = 0

        while True:  # there are still nodes we haven't found
            connected_counter += 1

            # evaluate halting criteria
            if not to_connect - visited_set:  # we visited all of the nodes we needed to search
                self.connected_true_counter.append(connected_counter)
                return True
            if not to_search:  # no more nodes available to search
                # raise ValueError("")
                self.connected_false_counter.append(connected_counter)
                return False

            next_node = to_search.pop()

            # new nodes that we haven't seen yet
            new_neighbors = set(self.graph.neighbors(to_search.pop())).intersection(district_nodes).difference(
                visited_set)

            priority_neighbors = to_connect.intersection(new_neighbors)
            nonpriority_neighbors = new_neighbors - priority_neighbors
            visited_set.update(new_neighbors)  # seen new neighbors
            to_search.extend(nonpriority_neighbors)
            to_search.extend(priority_neighbors)  # ensure that priority neighbors are at the top of the queue

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def handle_move(self, move):
        self.move_log.append(move)
        self.iteration += 1 # always change this
        if move is not None:
            # self.state_log.append(move)
            self.flip(move[0], move[2]) # TODO check this indexing


    def flip(self, node_id, district_id):
        # TODO this should be an internal method, since if you start executing it it won't update _proposal_state as well
        # update lookups
        old_color = self.node_to_color[node_id]
        self.color_to_node[old_color].remove(node_id)
        self.color_to_node[district_id].add(node_id)
        self.node_to_color[node_id] = district_id

        # record the move


        # for stat in self.tallied_stats:
        #     # the stat tally giveth
        #     self.stat_tally[stat][district_id] += self.graph.nodes()[node_id][stat] # node_data[node_id][stat]
        #     # ...and it taketh away
        #     self.stat_tally[stat][old_color] -= self.graph.nodes()[node_id][stat] # self.node_data[node_id][stat]

    # shim to add in data

    @classmethod
    def from_file_hardcode(cls):
        # TODO update this for new graph stuff
        # just for testing purposes on Duplin Onslow
        filepath = 'nonreversiblecodebase/data/DuplinOnslow/Duplin_C_Onslow_P__BORDERLENGTHS.txt'
        # todo sort out

        edges = []
        node_data = dict()
        delim = '\t'
        border_key = '-1'
        with open(filepath) as f:
            for line in f:
                L = line.strip().split(delim)
                if L[0] != border_key and L[1] != border_key:  # ignore border for now
                    edges.append((int(L[0]), int(L[1])))

        population_filepath = 'nonreversiblecodebase/data/DuplinOnslow/Duplin_C_Onslow_P__POPULATION.txt'
        with open(population_filepath) as f:
            for line in f:
                L = line.strip().split(delim)
                node_id = int(L[0])
                population = float(L[1])
                node_data[node_id] = {'population': population, 'node_count': 1}  # instantiate with population

        centroids_filepath = 'nonreversiblecodebase/data/DuplinOnslow/Duplin_C_Onslow_P__CENTROIDS.txt'
        with open(centroids_filepath) as f:
            for line in f:
                L = line.strip().split(delim)
                node_id = int(L[0])
                node_data[node_id]['x'] = float(L[1])
                node_data[node_id]['y'] = float(L[2])

        # node to color - hardcoded to
        coloring = {0: 1001, 1: 1001, 10: 1001, 19: 1001, 20: 1001,
                    2: 1002, 12: 1002, 21: 1002, 22: 1002, 23: 1002,
                    3: 1003, 9: 1003, 7: 1003, 5: 1003, 6: 1003,
                    4: 1004, 14: 1004, 15: 1004, 16: 1004, 17: 1004, 24: 1004,
                    8: 1005,  # just leave this one as a single to see if that causes issues
                    11: 1006, 13: 1006, 18: 1006}

        node_list = list(coloring.keys())

        return State.from_edge_node_list(node_list, edges, coloring=coloring, node_data=node_data)


    @classmethod
    def from_edge_node_list(cls, node_list, edge_list, coloring, node_data=None, edge_data=None, tallied_stats=('population', 'node_count')):
        ''' allows user to initialize state as (V,E), as well as node_data, edge_data instead of passing a nx.graph object with info attached
        user must still provide coloring as before'''

        graph = nx.Graph()
        graph.add_nodes_from(node_list)
        graph.add_edges_from(edge_list)

        if node_data is not None:
            nx.set_node_attributes(graph, node_data)
        if edge_data is not None:
            nx.set_edge_attributes(graph, edge_data)

        return State(graph, coloring, tallied_stats=tallied_stats)

    @classmethod
    def from_state(cls, state, **kwargs):
        '''Shim that accepts 'legacy' state object and initializes new class '''
        coloring = state['nodeToDistrict']
        graph = state['graph'] # includes all necessary data? do we need to recode anything?

        return State(graph, coloring, tallied_stats=[], **kwargs) # should tallied stats be different?



def greedy_graph_coloring(graph, num_districts=3):

    nodes_to_draft = set(graph.nodes())

    # initialize with random set
    seed_nodes = random.sample(nodes_to_draft, k=num_districts)
    for node_id in seed_nodes:
        nodes_to_draft.remove(node_id)
    color_to_node = {district_id: {seed_nodes.pop()} for district_id in range(num_districts)} # TODO these need to be actually random
    draftable_nodes = {district_id: set([i for i in itertools.chain(*[graph.neighbors(j) for j in color_to_node[district_id]])]) for district_id in color_to_node}
    drafted_nodes = set(itertools.chain(*color_to_node.values()))

    while nodes_to_draft:

        for district_id in color_to_node:
            while draftable_nodes[district_id]:
                node = draftable_nodes[district_id].pop()
                if node not in drafted_nodes:
                    drafted_nodes.add(node)
                    color_to_node[district_id].add(node)
                    draftable_nodes[district_id].update(graph.neighbors(node))
                    nodes_to_draft.remove(node)
                    break

    return color_to_node


def log_contested_edges(state):
    # maybe manage an array so we don't need to use these horrible loops
    for edge in state.contested_edges:
        # contested_nodes = set(itertools.chain(*state.contested_edges)) # this is horrible, maybe start tracking contested nodes?
        state.contested_edge_counter[edge] += 1

    for node in state.contested_nodes:
        state.contested_node_counter[node] += 1


def connected_breadth_first(state, node_id, old_color):

    district_nodes = state.color_to_node[old_color].difference({node_id}) # all of the old_color nodes without the one we're removing
    to_connect = district_nodes.intersection(set(state.graph.neighbors(node_id))) # need to show each of these are still connected

    if not to_connect: # TODO should this check get run here or should it be guaranteed?
        return False # we deleted the last node of a district, this is illegal

    init = to_connect.pop()
    to_search = [init] # ordered list of nodes available to search
    visited_set = {init}

    connected_counter = 0

    while True: # there are still nodes we haven't found
        connected_counter += 1

        # evaluate halting criteria
        if not to_connect - visited_set: # we visited all of the nodes we needed to search
            state.connected_true_counter.append(connected_counter)
            return True
        if not to_search: # no more nodes available to search
            # raise ValueError("")
            state.connected_false_counter.append(connected_counter)
            return False

        # new nodes that we haven't seen yet
        new_neighbors = set(state.graph.neighbors(to_search.pop())).intersection(district_nodes).difference(visited_set)

        priority_neighbors = to_connect.intersection(new_neighbors)
        nonpriority_neighbors = new_neighbors - priority_neighbors
        visited_set.update(new_neighbors) # seen new neighbors
        to_search.extend(nonpriority_neighbors)
        to_search.extend(priority_neighbors) # ensure that priority neighbors are at the top of the queue


class RingPointer(object):

    def __init__(self, node_id, downstream):
        self.node_id = node_id
        self.downstream = downstream # next ringpointer in chain


    def __hash__(self):
        return self.node_id # this is a compact hash


def ring_connectivity_naive(state):
    # create the ring connectivity objects for a thing
    # requires that edges be counted in a clockwise fashion - how to do this best?
    pass






def update_ring_connectivity(state):

    if not hasattr('ring_lookup', state):
        state.ring_lookup = ring_connectivity_naive(state)


    else:
        for move in state.move_log[state.ring_lookup_updated:]:
            if move is not None:
                node_id, old_color, new_color = move



                # todo stuff here



    state.ring_lookup_updated = state.iteration

def test_ring_connectivity(node_id, old_color, new_color, state):
    # test if number of neighbors is

    return len(state.ring_lookup[old_color][node_id].out_connections) == 1 # if there are multiple, we have a problem and can't remove this

