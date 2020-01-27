import logging
import networkx as nx
from collections import namedtuple, defaultdict
import numpy as np

## Need to
# Node = namedtuple('Node', ['node_id','coloring', 'x','y']) # simple struct for keeping track of nodes

# Remaining to-dos
# "Replay" function to parse state log and produce graph state at each step
# Sort out
# state initializer
#


####### Meeting notes 2019-12-06:
# Make 'state' with listeners that track need for metadata

# state object that allows arbitrary extensions
class State(object):

    def __init__(self, edges, coloring, node_data=None, edge_data=None, tallied_stats=('node_count', 'population')):
        # nodes are node ids, edges iterable of (node_id, node_id), coloring is (node_id, district_id)
        # node_id, district_id must be orderable, hashable
        # node_data, edge_data are optional, {node_id: dict} lookups
        # tallied stats will be updated after each iteration
        self.__dict__ = {}

        self.node_data = node_data # this is unchecked at present
        self.edge_data = edge_data # make sure this is easily accessible

        nodes = {edge[0] for edge in edges}.union({edge[1] for edge in edges})
        if nodes.difference(set(coloring.keys())) or set(coloring.keys()).difference(nodes):
            raise ValueError("Edges and coloring must match")

        self.graph = nx.Graph().add_edges_from(edges) # hopefully this will occur lazily
        self.node_to_color = coloring
        d = defaultdict(set)
        for node_id, district_id in coloring.items():
            d[district_id].add(node_id)
        self.color_to_node = dict(d)

        # frequently we need to keep track of , we have a special way of doing that
        self.stat_tally = defaultdict(dict)
        for district_id, nodes in self.color_to_node.items():
            for stat in tallied_stats:
                self.stat_tally[stat][district_id] = sum(self.node_data[node_id][stat] for node_id in nodes)

        self.tallied_stats = tallied_stats

        self.state_log = [] # contains (node_id, district_id) pairs of moves, in order
        self.iteration = 0 # this will get bumped each time we make a move


    def __setattr__(self, key, value):
        self.__dict__[key] = value


    def flip(self, node_id, district_id):

        # update lookups
        old_color = self.node_to_color[node_id]
        self.color_to_node[old_color].remove(node_id)
        self.color_to_node[district_id].add(node_id)
        self.node_to_color[node_id] = district_id

        # record the move
        self.state_log.append((node_id, district_id))

        self.iteration += 1 # add to iteration

        for stat in self.tallied_stats:
            # the stat tally giveth
            self.stat_tally[stat][district_id] += self.node_data[node_id][stat]
            # ...and it taketh away
            self.stat_tally[stat][old_color] -= self.node_data[node_id][stat]



def naive_init_flow(node_dict, edge_list, centroid):

    # generator
    # accepts node_dict, edge_list, reoutputs in correct directed order u -> v
    # edge_list tuple u -> v
    # centroid is tuple (x,y) for flow. flow is always CCW .
    # TODO possibly allow for elliptical flow?

    for edge in edge_list:
        n1 = node_dict[edge[0]]# the centerpoint between the two nodes
        n2 = node_dict[edge[1]]

        theta1 = np.math.atan2(n1.y-centroid[1], n1.x-centroid[0])
        theta2 = np.math.atan2(n2.y - centroid[1], n2.x - centroid[0])
        yield (edge if theta1 >= theta2 else (edge[1], edge[0])) # TODO the logical is wrong, doesn't account for branch cut issue

class LoggingMixin(object):

    def __init__(self, logfile=__name__):
        self.logger = logging.getLogger(logfile)

    def log(self):
        self.logger.info(self.state)

class ExtendableState(object):
    # TODO add way to deactivate listeners as needed

    def __init__(self):
        self._listeners = dict() # maybe we should expose as method calls?
        self.node_to_color = dict() # dict of int:int
        self.color_to_node = dict() # dict of int:set

    def update_state(self, prop):

        # prop is (node_id, district_id)
        # do basic graph update
        self.color_to_node[prop[1]].add()
        self.node_to_color[prop[0]] = prop[1]

        for listener in self._listeners:
            listener(self) # pass in State object, perform side-effecting operation

    def get_state(self):
        pass

    # need to check if registered 
    # if not registered, then register + instantiate
    # listeners should be abstracted from a class, one to many, and shareable among different classes
    # but also sandboxed from each other


    # state will ALWAYS have some node-to-color, color-to-node, adjacency matrix
    # state will be extensible with

    # attach to listeners
    # keep last_updated,


class PrecinctFlow(LoggingMixin):

    def __init__(self, node_dict, edge_list, coloring, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # node list a list of node_ids (hashable)
        # edge_list a list of 2-tuples (edge_id, edge_id) in initial direction
        # coloring a dict[node_id, color_id]

        self.graph = nx.Graph()
        self.graph.add_nodes_from(node_dict.values()) # TODO refactor
        self.graph.add_edges_from(edge_list)
        self.involution = [1] # keeping this as list to make more extensible in future

        # keep two lookups to keep both fast - maybe we'll get rid of one later
        dd = defaultdict(list)
        for k, v in node_dict.items():
            dd[v.coloring].append(k)
        self.color_to_nodes =  dict(dd)
        self.node_to_color = {k:v.coloring for k,v in node_dict.items()}

        self.contested_edges = self.contested_edges_naive() # keep a set of contested edge tuples (u,v)
        # should be undirected, always stored as (lesserNodeId, greaterNodeId)
        # self.involution_state

        self.state_log = [] # will store a list of accepted proposals, plus None if we reject/involve
        # we can replay this to establish the state at any given step, though it's gonna be a pain in the butt to write



    def proposal_distribution(self, prop):
        pass
        # J_u to evaluate the odds of proposal being accepted - this should be very customizeable
        # sometimes a call to involve will modify this func or its internals

    def accept_reject(self, prop):
        # TODO rng here
        # returns a boolean
        u = np.random.uniform()
        r = self.proposal_distribution(prop)

        return u < min(r, 1)


    # TODO ask about how involutions should work - always involve only one part at a time? handle multiple?
    def involve(self, idx):
        self.involution[idx]*=-1

    def mh_steps(self, n=1000):
        for i in range(n):
            self.step()
            self.log()

    def step(self):

        props, scores = self.get_proposals() # no side effects here, should be totally based on current state
        prop, score = self.pick_proposal(props, scores) # picked a proposal
        if self.accept_reject(prop):
            # proposal accepted!

            self.update_graph(prop) # sort out all of the backend stuff we need
            self.state_log.append(prop)

        else:
            # proposal rejected, conduct involution
            self.involve(0)
            # self.graph = self.graph.reverse()
            self.state_log.append(None) # just so we know what happened

    def update_graph(self, prop):
        # maintain the graph state after we accept a proposal

        oldcolor = self.node_to_color[prop[0]]
        newcolor = self.node_to_color[prop[1]]

        self.node_to_color[prop[0]] = newcolor
        self.color_to_nodes[oldcolor].remove(prop[0]) # is doing this going to be expensive for large precinct? No reason these can't be sets if it gets big
        self.color_to_nodes[newcolor].append(prop[0])

        # TODO check very carefully that this is doing the right thing - is i[1] always going to be out from original node?
        new_cont_edges = {i for i in self.graph.edges(prop[0]) if i[1] not in self.color_to_nodes[newcolor] }
        remove_cont_edges =  set(self.graph.edges(prop[0]))-new_cont_edges # if it's not contested, it should be removed

        # add new edges, remove the old ones
        self.contested_edges.update(new_cont_edges)
        self.contested_edges.difference_update(remove_cont_edges)


    def scores_to_probs(self, scores):
        # take scores, use probability distribution, normalize
        # currently uses an exponential kernel
        scores = 1./np.exp(scores)
        probs = scores/sum(scores) # make a probability distribution by normalizing that ish

        return probs

    def graph_boundary_naive(self, node_list):
        # for a given node_list, calculate the boundary size - this is an edge attribute. assumes this is kept
        # might be able to calculate this efficiently by intersecting with contested edges

        pass

    # TODO ask how we'd like this to work
    def score_proposal(self, proposed_smaller, proposed_bigger):
        pass


    def get_involution_strategy(self, prop):
        # for precinct-to-precint, we only have one involution index
        # this is just here so we can put in more complex logic/lookups for other algorithms
        # this has GOT to calculate it with the "canonical" edge ordering
        return 0


    def get_proposals(self):
        # get a scored list of proposals for consideration

        proposals = {}

        # alternative would be generate all 'articulation points' for each district
        #  then test if each edge[0] in contested_edges is articulation point

        for edge in self.graph.edges:
            if (min(edge), max(edge)) in self.contested_edges: # do we have to do min/max, or will it keep it consistent?

                # make a copy of edge with the correct ordering
                iedge = (edge[1], edge[0]) if self.involution[self.get_involution_strategy(edge)] else edge
                # TODO True/False would be cleaner than 1 vs. -1 and much more intuitive

                oldcolor = self.node_to_color[iedge[0]] # find the district that the original belongs to
                proposed_smaller = self.graph.subgraph([i for i in self.color_to_nodes[oldcolor] if i!=iedge[0]]) # the graph that lost a node
                proposed_bigger = self.graph.subgraph(self.color_to_nodes[iedge[1]] + [iedge[0]])
                if not nx.is_connected(proposed_smaller):
                    continue # can't disconnect districts

                # TODO population constraint enforcement - optional, min/max check
                # TODO constraint on compactness -
                # TODO constraint on traversals - figure out later

                score = self.score_proposal(proposed_smaller, proposed_bigger) # smaller score is better

                proposals[edge] = score # no normalization here

                # check that we won't wind up with a non-simply connected component (like one district surrounds another)


        # look at all possible candidates, return list of candidate proposals with associated weights

        return list(proposals.keys()), list(proposals.values()) # we need to be able to index these later

    def pick_proposal(self, proposals, scores):
        # TODO rng goes here
        weights = self.scores_to_probs(scores)
        idx = np.random.choice(range(len(weights)), p=weights)
        return proposals[idx], weights[idx]

def compute_com(state, district_id):
    # TODO we need this?
    pass


def center_of_mass_updater(func):
    # ensure that state has an updated copy of center of mass before this function runs
    def inner(state, *args, **kwargs):
        if not hasattr(state, 'district_to_com'):
            state.district_to_com = dict() # TODO fix this to init correctly
            state.district_to_com_updated = state.iteration


        if state.iteration != state.district_to_com_updated:
            districts_to_update = {district_id for node_id, district_id in state.state_log[state.district_to_com_updated:]}
            for district_id in districts_to_update:
                state.district_to_com[district_id] = compute_com(state, district_id)
        return func(state, *args, **kwargs)
    return inner

def contested_edges_naive(graph):
    # generate contested edges by testing each edge in the graph. it's brute force and definitely works
    contested = set()

    for edge in graph.edges:
        if graph.nodes[edge[0]]['coloring'] != graph.nodes[edge[1]]['coloring']:
            contested.add((min(edge), max(edge))) # always store small, large
    return contested


def contested_edges_updater(func):

    # before performing function, checks that 'state' contains an updated copy, and updates as needed.

    def inner(state, *args, **kwargs):
        # init to current state
        if not hasattr(state, 'contested_edges'):
            state.contested_edges = contested_edges_naive(state.graph)
            state.contested_edges_updated = state.iteration # set to current iteration

        # this may be an empty list if it's already been updated
        for node_id, district_id in state.state_log[state.contested_edges_updated:]:
            # move is provided as (node_id, color_id)
            neighbors = state.graph.edges(node_id)
            # edges to add
            state.contested_edges.update({(min(u,v), max(u,v)) for u, v in neighbors if state.node_to_color[v]!=district_id})
            # edges to remove
            state.contested_edges.difference_update({(min(u,v), max(u,v)) for u, v in neighbors if state.node_to_color[v]==district_id})

        #     # at some point it will be more efficient to just naively reconstruct the contested edges, we should look out for this
            state.contested_edges_updated = state.iteration




        # note that func must accept state as the FIRST argument. will this impact our ability to chain these together?
        return func(state, *args, **kwargs)

    return inner



