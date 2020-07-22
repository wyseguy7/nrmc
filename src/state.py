import collections
import networkx as nx
import pandas as pd
import os
import numpy as np
import itertools
import random

# from src.core import cython_biconnected
from .scores import population_balance_score # needed to keep population deviation updated

CENTROID_DIM_LENGTH = 2 # TODO do we need a settings.py file?
try:
    from biconnected import calculate_com_inner
    cython_biconnected = True
except ImportError:
    cython_biconnected = False




class State(object):

    def __init__(self, graph, coloring, log_contested_edges = True,
                 coerce_int = True, apd=0.1, ideal_pop = None, involution = 1):


        if coerce_int:
            # this is important to ensure it's compatible with our Cython speedups
            relabeling = {i: int(i) for i in graph.nodes()}
            coloring = {int(k):v for k,v in coloring.items()}
            graph = nx.relabel_nodes(graph, relabeling, copy=True)

        self.coerce_int = coerce_int
        self.graph = graph

        self.node_to_color = coloring
        d = collections.defaultdict(set)
        for node_id, district_id in coloring.items():
            d[district_id].add(node_id)
        self.color_to_node = dict(d)

        num_districts = len(self.color_to_node)
        minimum_population = len(self.graph.nodes())/num_districts - len(self.graph.nodes())*apd/2
        maximum_population = len(self.graph.nodes())/num_districts + len(self.graph.nodes())*apd/2

        self.minimum_population = minimum_population
        self.maximum_population = maximum_population
        self.iteration = 0
        self.state_log = []
        self.move_log = [] # move log is state_log plus 'None' each time we make an involution
        self.ideal_pop = ideal_pop
        self.involution = involution

        self.log_contested_edges = log_contested_edges # rip these out into a mixin?
        self.contested_edge_counter = collections.defaultdict(int)
        self.contested_node_counter = collections.defaultdict(int)


        self.connected_true_counter = [] # TODO remove after finished debugging
        self.connected_false_counter = []

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
                graph.add_node(other_node, boundary=True) # need to make sure we get boundary correct here


        # guarantee we have a boundary entry for each
        for node_id in graph.nodes():

            graph.nodes()[node_id]['Centroid'] = np.array([centroids_dict[node_id]['x'],
                                                           centroids_dict[node_id]['y']] , dtype='d')

            graph.nodes()[node_id]['population'] = pop_dict[node_id]['population']
            graph.nodes()[node_id]['area'] = area_dict[node_id]['area']


            if 'boundary' not in graph.nodes()[node_id]:
                graph.nodes()[node_id]['boundary'] = False

        minimum_population = len(graph.nodes())/num_districts - len(graph.nodes())*apd/2
        maximum_population = len(graph.nodes())/num_districts + len(graph.nodes())*apd/2

        # loop until we achieve a valid coloring
        while True:
            valid_coloring = True
            coloring = greedy_graph_coloring(graph, num_districts=num_districts)
            # check minimum population is valid
            for color, nodes in coloring.items():
                if sum(graph.nodes()[node_id]['population'] for node_id in nodes) < minimum_population or (
                    sum(graph.nodes()[node_id]['population'] for node_id in nodes) > maximum_population):
                    valid_coloring = False
                    break

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


def contested_edges_naive(state):
    # generate contested edges by testing each edge in the graph. it's brute force and definitely works
    contested = set()

    for edge in state.graph.edges:
        if state.node_to_color[edge[0]] != state.node_to_color[edge[1]]:
            contested.add((min(edge[0], edge[1]), max(edge[0], edge[1])))  # always store small, large
        # if state.graph.nodes[edge[0]]['coloring'] != graph.nodes[edge[1]]['coloring']:
        #     contested.add((min(edge), max(edge))) # always store small, large
    return contested


def calculate_com_naive(state, weight_attribute=None):
    com = {}
    total_weight = {}
    for district_id, nodes in state.color_to_node.items():
        if weight_attribute is None:
            weights = {node_id: 1 for node_id in nodes}
        else:
            weights = {state.graph.nodes()[node_id][weight_attribute] for node_id in nodes}

        total_weight[district_id] = sum(weights.values())
        com[district_id] = np.array([sum([state.graph.nodes()[i]['Centroid'][j]*weights[i] for i in nodes]
                                         )/total_weight[district_id] for j in range(CENTROID_DIM_LENGTH)], dtype='d')
    return com, total_weight


def update_center_of_mass(state):

    if not hasattr(state, 'com_centroid'):
        state.com_centroid, state.com_total_weight = calculate_com_naive(state)
        state.com_updated = state.iteration

    for i in state.move_log[state.com_updated:]:
        if i is not None:
            node_id, old_color, new_color = i
            (state.com_centroid[new_color], state.com_centroid[old_color],
             state.com_total_weight[new_color], state.com_total_weight[old_color]) = calculate_com_one_step(state, i)

    state.com_updated = state.iteration



def create_district_boundary_naive(state):

    district_boundary = collections.defaultdict(set)

    for node_id, color in state.node_to_color.items():

        for neighbor in state.graph.neighbors(node_id):
            if state.node_to_color[neighbor]!= color:
                neighbor_color = state.node_to_color[neighbor]
                district_boundary[(min(color, neighbor_color), max(color, neighbor_color))].add(
                    (min(node_id, neighbor), max(node_id, neighbor)))

    return district_boundary



def process_boundary_move(state, node_id, old_color, new_color, neighbor):
    neighbor_color = state.node_to_color[neighbor]
    if neighbor_color != new_color:
        key = (min(neighbor_color, new_color), max(neighbor_color, new_color))
        state.district_boundary[key].add((min(neighbor, node_id), max(neighbor, node_id)))
    if neighbor_color != old_color:
        key = (min(neighbor_color, old_color), max(neighbor_color, old_color))
        state.district_boundary[key].remove((min(neighbor, node_id), max(neighbor, node_id)))


def update_district_boundary(state):

    if not hasattr(state, 'district_boundary'):

        state.district_boundary = create_district_boundary_naive(state)
        state.district_boundary_updated = state.iteration

    moves_to_do = [i for i in state.move_log[state.district_boundary_updated:] if i is not None]

    # TODO we have repeated code in two places, rip out into function

    if len(moves_to_do) < 5:
        for node_id, old_color, new_color in moves_to_do:
            for neighbor in state.graph.neighbors(node_id):
                neighbor_color = state.node_to_color[neighbor]
                key = (min(neighbor_color, new_color), max(neighbor_color, new_color))
                # node_key = (min(neighbor, node_id), max(neighbor))

                if neighbor_color != new_color:
                    state.district_boundary[key].add((min(neighbor, node_id), max(neighbor, node_id)))
                else: # color == new_color
                    node_key = (min(neighbor, node_id), max(neighbor, node_id))
                    key = (min(neighbor_color, old_color), max(neighbor_color, old_color))
                    if node_key in state.district_boundary[key]:
                        state.district_boundary[key].remove(node_key)

    else:
        perturbed_nodes = dict()
        for node_id, old_color, new_color in moves_to_do:

            if node_id in perturbed_nodes:
                perturbed_nodes[node_id][1] = new_color
            else:
                perturbed_nodes[node_id] = (old_color, new_color)

        for node_id, (old_color, new_color) in perturbed_nodes.items():

            for neighbor in state.graph.neighbors(node_id):
                neighbor_color = state.node_to_color[neighbor]
                key = (min(neighbor_color, new_color), max(neighbor_color, new_color))

                if neighbor_color != new_color:
                    state.district_boundary[key].add((min(neighbor, node_id), max(neighbor, node_id)))
                else: # color == new_color
                    # key = (min(neighbor_color, new_color), max(neighbor_color, old_color))
                    state.district_boundary[key].remove((min(neighbor, node_id), max(neighbor, node_id)))

    state.district_boundary_updated = state.iteration

def update_boundary_nodes_naive(state):
    counter = collections.Counter([v for k,v in state.node_to_color.items() if state.graph.nodes()[k]['boundary']])
    return counter


def update_boundary_nodes(state):

    if not hasattr(state, 'boundary_node_counter'):
        state.boundary_node_counter = update_boundary_nodes_naive(state)

    else:
        for move in state.move_log[state.boundary_node_updated:]:
            if move is not None and state.graph.nodes()[move[0]]['boundary']:
                # if we flipped a boundary node
                state.boundary_node_counter[move[1]] -= 1
                state.boundary_node_counter[move[2]] += 1

    state.boundary_node_updated = state.iteration



def simply_connected(state, node_id, old_color, new_color):
    # asssuming that a proposal will result in a graph being connected, check that it will be simply connected
    # this occurs iff there is another node district:old_color that touches either boundary or another color
    # update_contested_edges(state) # do we need this here or can we pull it somewhere else?

    if state.boundary_node_counter[old_color] > 0:
        return True # this handles the vast majority of cases

    smaller = state.color_to_node[old_color] - {node_id}
    contested_nodes = {i[0] for i in state.contested_edges}.union({i[1] for i in state.contested_edges}).intersection(
        smaller) # nodes that are contested and are in the shrunk district
    for node in contested_nodes:
        # is this node either on the boundary, or does it touch some other color than
        if state.graph.nodes()[node]['boundary'] or len([i for i in state.graph.neighbors(node)
                                                         if state.node_to_color[i] not in (old_color, new_color)]) > 0:

            # about to return True - should be able to check that

            return True
    return False

def calculate_population_naive(state):
    return {district_id: sum(state.graph.nodes()[node_id]['population']
                             for node_id in state.color_to_node[district_id]) for district_id in state.color_to_node}




def contested_nodes_naive(state):
    contested = set()
    for node_id in state.graph.nodes():
        color = state.node_to_color[node_id]
        if not all(color==state.node_to_color[other_id] for other_id in nx.neighbors(state.graph, node_id)):
            contested.add(node_id)
    return contested


def log_contested_edges(state):
    # maybe manage an array so we don't need to use these horrible loops
    for edge in state.contested_edges:
        # contested_nodes = set(itertools.chain(*state.contested_edges)) # this is horrible, maybe start tracking contested nodes?
        state.contested_edge_counter[edge] += 1

    for node in state.contested_nodes:
        state.contested_node_counter[node] += 1


def update_contested_edges(state):
    if not hasattr(state, 'contested_edges'):
        state.contested_edges = contested_edges_naive(state)
        state.contested_nodes = contested_nodes_naive(state) # maybe we should track this separately?
        state.contested_edges_updated = state.iteration  # set to current iteration

    # this may be an empty list if it's already been updated
    for move in state.move_log[state.contested_edges_updated:]:

        if move is not None:
            node_id, old_color, new_color = move
            # move is provided as (node_id, color_id)
            neighbors = state.graph.edges(node_id)
            # edges to add
            state.contested_edges.update(
                {(min(u, v), max(u, v)) for u, v in neighbors if state.node_to_color[v] != new_color})
            # edges to remove
            state.contested_edges.difference_update(
                {(min(u, v), max(u, v)) for u, v in neighbors if state.node_to_color[v] == new_color})

            neighboring_nodes = set(nx.neighbors(state.graph, node_id))

            # add neighbors that aren't new_color
            state.contested_nodes.update(neighboring_nodes-state.color_to_node[new_color])

            # remove neighbors that are new_color
            state.contested_nodes.difference_update(neighboring_nodes.intersection(state.color_to_node[new_color]))

        if state.log_contested_edges:
            log_contested_edges(state) # separated into function so we can track how expensive this is

    #     # at some point it will be more efficient to just naively reconstruct the contested edges, we should look out for this
    state.contested_edges_updated = state.iteration


def perimeter_naive(state):
    # TODO refactor
    dd = collections.defaultdict(int)

    for n0, n1 in state.contested_edges:
        shared_length = state.graph.edges()[(n0,n1)]['border_length']
        dd[state.node_to_color[n0]] += shared_length
        dd[state.node_to_color[n1]] += shared_length

    return dd


def update_perimeter_aggressive(state):

    # this version assumes that this will get run EVERY time a node is flipped
    update_contested_edges(state) # guarantee contested edges updated before proceeding

    if not hasattr(state, 'district_to_perimeter'):
        state.district_to_perimeter = perimeter_naive(state)
        state.perimeter_updated = state.iteration  # set to current iteration

    for move in state.move_log[state.perimeter_updated:]:

        if move is None:
            continue

        node_id, old_color, new_color = move

        for neighbor in state.graph.neighbors(node_id):
            if neighbor in state.color_to_node[new_color]:
                # we need to reduce the perimeter of new_color by their shared amount
                state.district_to_perimeter[new_color] -= state.graph.edges[(node_id, neighbor)]['border_length']

            elif neighbor in state.color_to_node[old_color]:
                # we need to increase the perimeter of old_color by their shared amount
                state.district_to_perimeter[old_color] += state.graph.edges[(node_id, neighbor)]['border_length']

            else:
                # we need to increase the perimeter of new_color AND decrease of old color. no change to the perimeter of the 3rd district
                state.district_to_perimeter[new_color] += state.graph.edges[(node_id, neighbor)]['border_length']
                state.district_to_perimeter[old_color] -= state.graph.edges[(node_id, neighbor)]['border_length']

    state.perimeter_updated = state.iteration



def update_population(state):

    if not hasattr(state, 'population_counter'):
        state.population_counter = calculate_population_naive(state)

        if state.ideal_pop is None:
            state.ideal_pop = sum(state.population_counter.values())/len(state.population_counter)

        state.population_deviation = population_balance_naive(state, state.ideal_pop)

    else:
        for move in state.move_log[state.population_counter_updated:]:
            if move is not None:
                node_id, old_color, new_color = move
                state.population_counter[old_color] -= state.graph.nodes()[node_id]['population']
                state.population_counter[new_color] += state.graph.nodes()[node_id]['population']
                state.population_deviation = population_balance_score(state, move)

    state.population_counter_updated = state.iteration

def check_population(state, node_id, old_color, new_color, minimum=400, maximum=1200):
     return ((state.population_counter[old_color] - state.graph.nodes()[node_id]['population']) > minimum
     and state.population_counter[new_color] + state.graph.nodes()[node_id]['population'] < maximum
     )

def population_balance_naive(state, ideal_pop):
    return np.sqrt(sum((sum([state.graph.nodes()[node]['population'] for node in nodes]) - ideal_pop)**2
                       for nodes in state.color_to_node.values()))




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


def naive_init_flow(node_dict, edge_list, centroid):
    # generator
    # accepts node_dict, edge_list, reoutputs in correct directed order u -> v
    # edge_list tuple u -> v
    # centroid is tuple (x,y) for flow. flow is always CCW .
    # TODO possibly allow for elliptical flow?

    for edge in edge_list:
        n1 = node_dict[edge[0]]  # the centerpoint between the two nodes
        n2 = node_dict[edge[1]]

        theta1 = np.math.atan2(n1.y - centroid[1], n1.x - centroid[0])
        theta2 = np.math.atan2(n2.y - centroid[1], n2.x - centroid[0])
        yield (edge if theta1 >= theta2 else (
            edge[1], edge[0]))  # TODO the logical is wrong, doesn't account for branch cut issue



def compactness_naive(state):

    perimeter_dict = perimeter_naive(state)

    area_dict = {district_id: sum(node['population'] for node in state.graph.nodes() if node in state.graph.color_to_node[district_id])
                 for district_id in state.graph.color_to_node}

    return sum(perimeter_dict[district_id]**2/area_dict[district_id] for district_id in state.graph.color_to_node)


def calculate_com_one_step(state, proposal, weight_attribute=None):
    node_id, old_color, new_color = proposal

    #     com_centroid = copy.deepcopy(state.com_centroid) # ugh, should this function just be side-effecting? how bad is this cost?
    #     total_weight = copy.deepcopy(state.com_total_weight)
    node = state.graph.nodes()[node_id]  # how expensive is this lookup, anyways?

    weight = node[weight_attribute] if weight_attribute is not None else 1

    if cython_biconnected:
        output_new = calculate_com_inner(node['Centroid'], weight, state.com_centroid[new_color],
                                         state.com_total_weight[new_color])
        output_old = calculate_com_inner(node['Centroid'], -weight, state.com_centroid[old_color],
                                         state.com_total_weight[old_color])

        return np.array(output_new[0:2], dtype='d'), np.array(output_old[0:2], dtype='d'), output_new[2], output_old[2]

    else:

        centroid_new_color = (node['Centroid'] * weight + state.com_centroid[new_color] * state.com_total_weight[new_color])/(
                state.com_total_weight[new_color] + weight)
        centroid_old_color = (-node['Centroid'] * weight + state.com_centroid[old_color] * state.com_total_weight[old_color])/(
                state.com_total_weight[old_color] - weight)

        total_weight_new_color = state.com_total_weight[new_color] + weight
        total_weight_old_color = state.com_total_weight[old_color] - weight

        return centroid_new_color, centroid_old_color, total_weight_new_color, total_weight_old_color