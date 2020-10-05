# distutils: language = c++

from libcpp.unordered_map cimport unordered_map
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libcpp.pair cimport pair
from libc.math cimport sqrt
from cpython.array cimport array

cdef int[2][2] ROT_MATRIX = [[0, -1], [1, 0]]


cdef struct StackInner:
    int grandparent
    int parent
    vector[int] children

cdef int fast_min(int a, int b):
    if a < b:
        return a
    else:
        return b

cdef long pack_int(int a, int b):
    cdef long response = a #
    return (response << 32) & b # hope this works correctly



cdef class PerimeterComputer:

    cdef unordered_map[int, vector[int]] adj_mapping_full
    cdef unordered_map[int, unordered_set[int]] node_to_color # will Cython tolerate this?

    # cdef unordered_map[(int, int), float] border_length
    cdef unordered_map[long, float] border_length_lookup # static
    cdef unordered_map[int, float] external_border_lookup # static


    def __init__(self, unordered_map[int, vector[int]] adj_mapping_full,
                                                       unordered_map[int, unordered_set[int]] node_to_color,
                                                      unordered_map[long, float] border_length_lookup,
                                                          unordered_map[int, float] external_border_lookup):
        self.adj_mapping_full = adj_mapping_full
        self.node_to_color = node_to_color
        self.border_length_lookup = border_length_lookup
        self.external_border_lookup = external_border_lookup


    def update(self, int node_id, int old_color, int new_color):
        # need to keep node_to_color up to date
        self.node_to_color[old_color].erase(node_id)
        self.node_to_color[new_color].insert(node_id)


    cpdef float compactness_score(self, double area_larger, double area_smaller, double area_node, int node_id,
        double perim_smaller, double perim_larger, int old_color, int new_color, bint use_external_border):

        cdef float border_length
        cdef int neighbor
        cdef ext_border # not always used - is this bad?

        cdef float area_larger_new = area_larger + area_node
        cdef float area_smaller_new = area_smaller - area_node

        cdef float perim_smaller_new = perim_smaller # initialize at other values
        cdef float perim_larger_new = perim_larger

        for i in range(self.adj_mapping_full[node_id].size()):

            neighbor = self.adj_mapping_full[node_id][i]

            # node_pair = (node_id, neighbor) # check that this isn't hitting Python - cast explicitly to pair if it is
            # cython doesn't implement unordered_map for Pair<int, int> so we have to pack into a long
            border_length = self.border_length_lookup[pack_int(node_id, neighbor)]

            if self.node_to_color[new_color].count(neighbor) > 0:
                perim_larger_new -= border_length
            elif self.node_to_color[old_color].count(neighbor) > 0:
                perim_smaller_new += border_length
            else:
                perim_larger_new += border_length
                perim_smaller_new -= border_length

        if use_external_border:
            ext_border = self.external_border_lookup[node_id]
            perim_larger_new += ext_border
            perim_smaller_new -= ext_border

        cdef float score_old = perim_smaller ** 2 / area_smaller + perim_larger ** 2 / area_larger
        cdef float score_new = perim_smaller_new ** 2 / (area_smaller - area_node) + perim_larger_new ** 2 / (area_larger + area_node)

        return score_new - score_old



cpdef float population_balance_sq(double flipped_pop, double new_pop, double old_pop):
    return 2 * flipped_pop * (new_pop - old_pop + flipped_pop)



cpdef float dot_product(double[:] a, double[:] b, double[:] center):

    cdef float vec_a_b[2]
    cdef float midpoint[2]
    cdef float vec_perp_center[2]
    cdef float vec_from_center[2]
    cdef float dp = 0.
    cdef float norm_center = 0.
    cdef float norm_a_b = 0.


    for i in range(2):
        vec_a_b[i] = b[i] - a[i]
        midpoint[i] = vec_a_b[i]*0.5 + a[i]
        vec_from_center[i] = midpoint[i] - center[i]

    vec_perp_center[0] = -1.0 * vec_from_center[1]
    vec_perp_center[1] = vec_from_center[0]

    for i in range(2):
        # dp += vec_a_b[i]*vec_from_center[i]
        dp += vec_a_b[i]*vec_perp_center[i]
        norm_center += vec_perp_center[i]**2
        norm_a_b += vec_a_b[i]**2

    return dp/sqrt(norm_center)/sqrt(norm_a_b)


cpdef calculate_com_inner(double[:] centroid, double weight,
                                                     double[:] com_centroid, double com_weight):

    # node_id, old_color, new_color = proposal

#     com_centroid = copy.deepcopy(state.com_centroid) # ugh, should this function just be side-effecting? how bad is this cost?
#     total_weight = copy.deepcopy(state.com_total_weight)
    # node = state.graph.nodes()[node_id] # how expensive is this lookup, anyways?

    # weight = node[weight_attribute] if weight_attribute is not None else 1
    cdef double[2] weighted_centroid


    weighted_centroid[0] = weight*centroid[0]
    weighted_centroid[1] = weight*centroid[1]
    # centroid[1] *= weight # now a weighted centroid

    cdef double[3] output # is this correct way of describing?

    output[2] = weight + com_weight

    for i in range(2):
        output[i] = (weighted_centroid[i] + com_centroid[i]*com_weight)/output[2]

    return output


# cpdef vector[int] biconnected_dfs(unordered_map[int, vector[int]] node_adj):
cpdef vector[int] biconnected_dfs(vector[int] node_list, unordered_map[int, vector[int]] node_adj):



    # depth-first search algorithm to generate articulation points
    # and biconnected components
    cdef unordered_set[int] visited
    cdef int start
    cdef unordered_map[int, int] discovery
    cdef unordered_map[int, int] low
    cdef int root_children = 0
    cdef vector[StackInner] stack
    # cdef StackInner last_el
    cdef vector[int] output # our output
    cdef StackInner last_el


    for i in range(node_list.size()):
    # for it in node_adj.begin():
        # start = it.first
        start = node_list[i]

        if visited.count(start) !=0:
            continue

        discovery.clear()
        low.clear()

        # TODO getting python useage here
        discovery[start] = 0
        low[start] = 0
        root_children = 0
        visited.insert(start)

        stack.clear()
        stack.push_back(StackInner(start, start, node_adj[start]))
        # stack.push_back(StackInner(start, start, it.second))
        while stack.size() > 0:
            last_el = stack.back() # will this work?
            grandparent = last_el.grandparent
            parent = last_el.parent
            children = last_el.children

            if children.size() > 0:
                child = children.back() # TODO make this work with pointers
                stack.back().children.pop_back() # will this have the requisite side effects?
                children.pop_back()

                if grandparent == child:
                    # print("p2")
                    continue

                if visited.count(child) !=0:
                    if discovery[child] <= discovery[parent]:  # back edge
                        low[parent] = fast_min(low[parent], discovery[child])
                else:
                    low[child] = discovery.size()
                    discovery[child] = discovery.size()
                    visited.insert(child)
                    stack.push_back(StackInner(parent, child, node_adj[child]))
            else:

                stack.pop_back()
                if stack.size() > 1:
                    if low[parent] >= discovery[grandparent]:
                        output.push_back(grandparent)
                    low[grandparent] = fast_min(low[parent], low[grandparent])
                elif stack.size() > 0:  # length 1 so grandparent is root
                    root_children += 1
            # root node is articulation point if it has more than 1 child
        if root_children > 1:
            output.push_back(start)
            #yield start

    return output
