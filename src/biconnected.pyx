# distutils: language = c++

from libcpp.unordered_map cimport unordered_map
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


cpdef dot_product(double[:] a, double[:] b, double[:] center):

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


# cpdef vector[int] mapping_test( unordered_map[int, vector[int]] node_adj):
#     for it in node_adj.begin():
#         print(it)

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
