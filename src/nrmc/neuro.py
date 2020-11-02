import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import softmax
from typing import List, Dict, Tuple



def estimate_target_distribution(probs: Dict, dim_t: int = 2) -> np.ndarray:
    """
    Estimate target distribution via the average of sorted source probabilities
    Args:
        probs: a dictionary of graphs {key: graph idx,
                                       value: (n_s, 1) the distribution of source nodes}
        dim_t: the dimension of target distribution
    Returns:
        p_t: (dim_t, 1) vector representing a distribution
    """
    p_t = np.zeros((dim_t, 1))
    x_t = np.linspace(0, 1, p_t.shape[0])
    for n in probs.keys():
        p_s = probs[n][:, 0]
        p_s = np.sort(p_s)[::-1]
        x_s = np.linspace(0, 1, p_s.shape[0])
        p_t_n = np.interp(x_t, x_s, p_s)
        p_t[:, 0] += p_t_n
    p_t /= np.sum(p_t)
    return p_t


def node_pair_assignment(trans: np.ndarray, p_s: np.ndarray, p_t: np.ndarray,
                         idx2node_s: Dict, idx2node_t: Dict) -> Tuple[List, List, List]:
    """
    Match the nodes in a graph to those of another graph
    Args:
        trans: (n_s, n_t) optimal transport matrix
        p_s: (n_s, 1) vector representing the distribution of source nodes
        p_t: (n_t, 1) vector representing the distribution of target nodes
        idx2node_s: a dictionary {key: idx of source node, value: the name of source node}
        idx2node_t: a dictionary {key: idx of target node, value: the name of target node}
    Returns:
        pairs_idx: a list of node index pairs
        pairs_name: a list of node name pairs
        pairs_confidence: a list of confidence of node pairs
    """
    pairs_idx = []
    pairs_name = []
    pairs_confidence = []
    if trans.shape[0] >= trans.shape[1]:
        source_idx = list(range(trans.shape[0]))
        for t in range(trans.shape[1]):
            column = trans[:, t] / p_s[:, 0]  # p(t | s)
            idx = np.argsort(column)[::-1]
            for n in range(idx.shape[0]):
                if idx[n] in source_idx:
                    s = idx[n]
                    pairs_idx.append([s, t])
                    pairs_name.append([idx2node_s[s], idx2node_t[t]])
                    pairs_confidence.append(trans[s, t])
                    source_idx.remove(s)
                    break
    else:
        target_idx = list(range(trans.shape[1]))
        for s in range(trans.shape[0]):
            row = trans[s, :] / p_t[:, 0]
            idx = np.argsort(row)[::-1]
            for n in range(idx.shape[0]):
                if idx[n] in target_idx:
                    t = idx[n]
                    pairs_idx.append([s, t])
                    pairs_name.append([idx2node_s[s], idx2node_t[t]])
                    pairs_confidence.append(trans[s, t])
                    target_idx.remove(t)
                    break
    return pairs_idx, pairs_name, pairs_confidence


def node_set_assignment(trans: Dict, probs: Dict, idx2nodes: Dict) -> Tuple[List, List, List]:
    """
    Match the nodes across two or more graphs according to their optimal transport to the barycenter
    Args:
        trans: a dictionary of graphs {key: graph idx,
                                       value: (n_s, n_c) optimal transport between source graph and barycenter}
               where n_s >= n_c for all graphs
        probs: a dictionary of graphs {key: graph idx,
                                       value: (n_s, 1) the distribution of source nodes}
        idx2nodes: a dictionary of graphs {key: graph idx,
                                           value: a dictionary {key: idx of row in cost,
                                                                value: name of node}}
    Returns:
        set_idx: a list of node index paired set
        set_name: a list of node name paired set
        set_confidence: a list of confidence set of node pairs
    """
    set_idx = []
    set_name = []
    set_confidence = []

    pairs_idx = {}
    pairs_name = {}
    pairs_confidence = {}
    num_sets = 0
    for n in trans.keys():
        source_idx = list(range(trans[n].shape[0]))
        pair_idx = []
        pair_name = []
        pair_confidence = []
        num_sets = trans[n].shape[1]
        for t in range(trans[n].shape[1]):
            column = trans[n][:, t] / probs[n][:, 0]
            idx = np.argsort(column)[::-1]
            for i in range(idx.shape[0]):
                if idx[i] in source_idx:
                    s = idx[i]
                    pair_idx.append(s)
                    pair_name.append(idx2nodes[n][s])
                    pair_confidence.append(trans[n][s, t])
                    source_idx.remove(idx[i])
                    break
        pairs_idx[n] = pair_idx
        pairs_name[n] = pair_name
        pairs_confidence[n] = pair_confidence

    for t in range(num_sets):
        correspondence_idx = []
        correspondence_name = []
        correspondence_confidence = []
        for n in trans.keys():
            correspondence_idx.append(pairs_idx[n][t])
            correspondence_name.append(pairs_name[n][t])
            correspondence_confidence.append(pairs_confidence[n][t])
        set_idx.append(correspondence_idx)
        set_name.append(correspondence_name)
        set_confidence.append(correspondence_confidence)
    return set_idx, set_name, set_confidence


def node_cluster_assignment(cost_s: csr_matrix, trans: np.ndarray, p_s: np.ndarray,
                            p_c: np.ndarray, idx2node: Dict) -> Tuple[Dict, Dict, Dict]:
    """
    Assign nodes of a graph to different clusters according to learned optimal transport
    Args:
        cost_s: a (n_s, n_s) adjacency matrix of a graph
        trans: a (n_s, n_c) optimal transport matrix, n_c is the number of clusters
        p_s: a (n_s, 1) vector representing the distribution of source nodes
        p_c: a (n_c, 1) vector representing the distribution of clusters
        idx2node: a dictionary {key: idx of cost_s's row, value: the name of node}
    Returns:
        sub_costs: a dictionary {key: cluster idx,
                                 value: a sub adjacency matrix of the sub-graph (cluster)}
        sub_idx2nodes: a dictionary {key: cluster idx,
                                     value: a dictionary {key: idx of sub-cost's row,
                                                          value: the name of node}}
        sub_probs: a dictionary {key: cluster idx,
                                 value: a vector representing distribution of subset of nodes}
    """
    cluster_id = {}
    sub_costs = {}
    sub_idx2nodes = {}
    sub_probs = {}

    for r in range(trans.shape[0]):
        row = trans[r, :] / p_c[:, 0]
        idx = np.argmax(row)
        # print(idx)
        if idx not in cluster_id.keys():
            cluster_id[idx] = [r]
        else:
            cluster_id[idx].append(r)

    for key in cluster_id.keys():
        indices = cluster_id[key]
        indices.sort()
        sub_costs[key] = cost_s[indices, :]
        sub_costs[key] = sub_costs[key][:, indices]
        sub_probs[key] = p_s[indices, :] / np.sum(p_s[indices, :])
        tmp_idx2node = {}
        for i in range(len(indices)):
            ori_id = indices[i]
            node = idx2node[ori_id]
            tmp_idx2node[i] = node
        sub_idx2nodes[key] = tmp_idx2node

    return sub_costs, sub_probs, sub_idx2nodes


def graph_partition(cost_s: csr_matrix, p_s: np.ndarray, p_t: np.ndarray,
                    idx2node: Dict, ot_hyperpara: Dict, trans0: np.ndarray=None) -> Tuple[Dict, Dict, Dict, np.ndarray]:
    """
    Achieve a single graph partition via calculating Gromov-Wasserstein discrepancy
    between the target graph and proposed one
    Args:
        cost_s: (n_s, n_s) adjacency matrix of source graph
        p_s: (n_s, 1) the distribution of source nodes
        p_t: (n_t, 1) the distribution of target nodes
        idx2node: a dictionary {key = idx of row in cost, value = name of node}
        ot_hyperpara: a dictionary of hyperparameters
    Returns:
        sub_costs: a dictionary {key: cluster idx,
                                 value: sub cost matrices}
        sub_probs: a dictionary {key: cluster idx,
                                 value: sub distribution of nodes}
        sub_idx2nodes: a dictionary {key: cluster idx,
                                     value: a dictionary mapping indices to nodes' names
        trans: (n_s, n_t) the optimal transport
    """
    cost_t = csr_matrix(np.diag(p_t[:, 0]))
    # cost_t = 1 / (1 + cost_t)
    trans, d_gw, p_s = gromov_wasserstein_discrepancy(cost_s, cost_t, p_s, p_t, ot_hyperpara, trans0)
    sub_costs, sub_probs, sub_idx2nodes = node_cluster_assignment(cost_s, trans, p_s, p_t, idx2node)
    return sub_costs, sub_probs, sub_idx2nodes, trans


def recursive_graph_partition(cost_s: csr_matrix, p_s: np.ndarray, idx2node: Dict, ot_hyperpara: Dict,
                              max_node_num: int = 200) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
    """
    Achieve recursive multi-graph partition via calculating Gromov-Wasserstein barycenter
    between the target graphs and a proposed one
    Args:
        cost_s: (n_s, n_s) adjacency matrix of source graph
        p_s: (n_s, 1) the distribution of source nodes
        idx2node: a dictionary {key = idx of row in cost, value = name of node}
        ot_hyperpara: a dictionary of hyperparameters
        max_node_num: the maximum number of nodes in a sub-graph
    Returns:
        sub_costs_all: a dictionary of graph {key: graph idx,
                                              value: a dictionary {key: cluster idx,
                                                                   value: sub cost matrices}}
        sub_idx2nodes: a dictionary of graph {key: graph idx,
                                              value: a dictionary {key: cluster idx,
                                                                   value: a dictionary mapping indices to nodes' names}}
        trans: (n_s, n_t) the optimal transport
        cost_t: the reference graph corresponding to partition result
    """
    costs_all = [cost_s]
    probs_all = [p_s]
    idx2nodes_all = [idx2node]
    costs_final = []
    probs_final = []
    idx2nodes_final = []
    n = 0
    while len(costs_all) > 0:
        costs_tmp = []
        probs_tmp = []
        idx2nodes_tmp = []
        for i in range(len(costs_all)):
            # print('Partition: level {}, leaf {}/{}'.format(n+1, i+1, len(costs_all)))
            p_t = estimate_target_distribution({0: probs_all[i]}, dim_t=2)
            # print(p_t[:, 0], probs_all[i].shape[0])
            cost_t = csr_matrix(np.diag(p_t[:, 0]))
            # cost_t = 1 / (1 + cost_t)
            ot_hyperpara['outer_iteration'] = probs_all[i].shape[0]
            trans, d_gw, p_s = gromov_wasserstein_discrepancy(costs_all[i],
                                                                  cost_t,
                                                                  probs_all[i],
                                                                  p_t,
                                                                  ot_hyperpara)
            sub_costs, sub_probs, sub_idx2nodes = node_cluster_assignment(costs_all[i],
                                                                          trans,
                                                                          probs_all[i],
                                                                          p_t,
                                                                          idx2nodes_all[i])

            for key in sub_idx2nodes.keys():
                sub_cost = sub_costs[key]
                sub_prob = sub_probs[key]
                sub_idx2node = sub_idx2nodes[key]
                if len(sub_idx2node) > max_node_num:
                    costs_tmp.append(sub_cost)
                    probs_tmp.append(sub_prob)
                    idx2nodes_tmp.append(sub_idx2node)
                else:
                    costs_final.append(sub_cost)
                    probs_final.append(sub_prob)
                    idx2nodes_final.append(sub_idx2node)

        costs_all = costs_tmp
        probs_all = probs_tmp
        idx2nodes_all = idx2nodes_tmp
        n += 1
    return costs_final, probs_final, idx2nodes_final


def multi_graph_partition(costs: Dict, probs: Dict, p_t: np.ndarray,
                          idx2nodes: Dict, ot_hyperpara: Dict,
                          weights: Dict = None,
                          predefine_barycenter: bool = False) -> \
        Tuple[List[Dict], List[Dict], List[Dict], Dict, np.ndarray]:
    """
    Achieve multi-graph partition via calculating Gromov-Wasserstein barycenter
    between the target graphs and a proposed one
    Args:
        costs: a dictionary of graphs {key: graph idx,
                                       value: (n_s, n_s) adjacency matrix of source graph}
        probs: a dictionary of graphs {key: graph idx,
                                       value: (n_s, 1) the distribution of source nodes}
        p_t: (n_t, 1) the distribution of target nodes
        idx2nodes: a dictionary of graphs {key: graph idx,
                                           value: a dictionary {key: idx of row in cost,
                                                                value: name of node}}
        ot_hyperpara: a dictionary of hyperparameters
        weights: a dictionary of graph {key: graph idx,
                                       value: the weight of the graph}
        predefine_barycenter: False: learn barycenter, True: use predefined barycenter
    Returns:
        sub_costs_all: a list of graph dictionary: a dictionary {key: graph idx,
                                                                 value: sub cost matrices}}
        sub_idx2nodes: a list of graph dictionary: a dictionary {key: graph idx,
                                                                 value: a dictionary mapping indices to nodes' names}}
        trans: a dictionary {key: graph idx,
                             value: an optimal transport between the graph and the barycenter}
        cost_t: the reference graph corresponding to partition result
    """
    sub_costs_cluster = []
    sub_idx2nodes_cluster = []
    sub_probs_cluster = []

    sub_costs_all = {}
    sub_idx2nodes_all = {}
    sub_probs_all = {}
    if predefine_barycenter is True:
        cost_t = csr_matrix(np.diag(p_t[:, 0]))
        trans = {}
        for n in costs.keys():
            sub_costs_all[n], sub_probs_all[n], sub_idx2nodes_all[n], trans[n] = graph_partition(costs[n],
                                                                                                 probs[n],
                                                                                                 p_t,
                                                                                                 idx2nodes[n],
                                                                                                 ot_hyperpara)
    else:
        cost_t, trans, _ = gromov_wasserstein_barycenter(costs, probs, p_t, ot_hyperpara, weights)
        for n in costs.keys():
            sub_costs, sub_probs, sub_idx2nodes = node_cluster_assignment(costs[n],
                                                                          trans[n],
                                                                          probs[n],
                                                                          p_t,
                                                                          idx2nodes[n])
            sub_costs_all[n] = sub_costs
            sub_idx2nodes_all[n] = sub_idx2nodes
            sub_probs_all[n] = sub_probs

    for i in range(p_t.shape[0]):
        sub_costs = {}
        sub_idx2nodes = {}
        sub_probs = {}
        for n in costs.keys():
            if i in sub_costs_all[n].keys():
                sub_costs[n] = sub_costs_all[n][i]
                sub_idx2nodes[n] = sub_idx2nodes_all[n][i]
                sub_probs[n] = sub_probs_all[n][i]
        sub_costs_cluster.append(sub_costs)
        sub_idx2nodes_cluster.append(sub_idx2nodes)
        sub_probs_cluster.append(sub_probs)

    return sub_costs_cluster, sub_probs_cluster, sub_idx2nodes_cluster, trans, cost_t


def recursive_multi_graph_partition(costs: Dict, probs: Dict, idx2nodes: Dict,
                                    ot_hyperpara: Dict, weights: Dict = None, predefine_barycenter: bool = False,
                                    cluster_num: int = 2, partition_level: int = 3, max_node_num: int = 200
                                    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Achieve recursive multi-graph partition via calculating Gromov-Wasserstein barycenter
    between the target graphs and a proposed one
    Args:
        costs: a dictionary of graphs {key: graph idx,
                                       value: (n_s, n_s) adjacency matrix of source graph}
        probs: a dictionary of graphs {key: graph idx,
                                       value: (n_s, 1) the distribution of source nodes}
        idx2nodes: a dictionary of graphs {key: graph idx,
                                           value: a dictionary {key: idx of row in cost,
                                                                value: name of node}}
        ot_hyperpara: a dictionary of hyperparameters
        weights: a dictionary of graph {key: graph idx,
                                       value: the weight of the graph}
        predefine_barycenter: False: learn barycenter, True: use predefined barycenter
        cluster_num: the number of clusters when doing graph partition
        partition_level: the number of partition levels
        max_node_num: the maximum number of nodes in a sub-graph
    Returns:
        sub_costs_all: a dictionary of graph {key: graph idx,
                                              value: a dictionary {key: cluster idx,
                                                                   value: sub cost matrices}}
        sub_idx2nodes: a dictionary of graph {key: graph idx,
                                              value: a dictionary {key: cluster idx,
                                                                   value: a dictionary mapping indices to nodes' names}}
        trans: (n_s, n_t) the optimal transport
        cost_t: the reference graph corresponding to partition result
    """
    num_graphs = len(costs)
    costs_all = [costs]
    probs_all = [probs]
    idx2nodes_all = [idx2nodes]
    costs_final = []
    probs_final = []
    idx2nodes_final = []
    n = 0
    while n < partition_level and len(costs_all) > 0:
        costs_tmp = []
        probs_tmp = []
        idx2nodes_tmp = []
        for i in range(len(costs_all)):
            # print('Partition: level {}, leaf {}/{}'.format(n+1, i+1, len(costs_all)))
            p_t = estimate_target_distribution(probs_all[i], cluster_num)
            # print(p_t[:, 0])

            max_node = 0
            for key in idx2nodes_all[i]:
                node_num = len(idx2nodes_all[i][key])
                if max_node < node_num:
                    max_node = node_num
            ot_hyperpara['outer_iteration'] = max([max_node, 200])

            sub_costs, sub_probs, sub_idx2nodes, _, _ = multi_graph_partition(
                costs_all[i], probs_all[i], p_t, idx2nodes_all[i], ot_hyperpara, weights, predefine_barycenter)

            for ii in range(len(sub_idx2nodes)):
                # print(len(sub_idx2nodes[ii]))
                if len(sub_idx2nodes[ii]) == num_graphs:
                    max_node = 0
                    for key in sub_idx2nodes[ii]:
                        node_num = len(sub_idx2nodes[ii][key])
                        # print('leaf {}, partition {}/{}, graph idx: {}, #node={}'.format(
                        #     i+1, ii+1, len(sub_idx2nodes), key, node_num))
                        if max_node < node_num:
                            max_node = node_num
                    if max_node > max_node_num:  # can be further partitioned
                        costs_tmp.append(sub_costs[ii])
                        probs_tmp.append(sub_probs[ii])
                        idx2nodes_tmp.append(sub_idx2nodes[ii])
                    else:
                        costs_final.append(sub_costs[ii])
                        probs_final.append(sub_probs[ii])
                        idx2nodes_final.append(sub_idx2nodes[ii])
        costs_all = costs_tmp
        probs_all = probs_tmp
        idx2nodes_all = idx2nodes_tmp
        n += 1

    if len(costs_all) > 0:
        costs_final += costs_all
        probs_final += probs_all
        idx2nodes_final += idx2nodes_all

    return costs_final, probs_final, idx2nodes_final


def direct_graph_matching(cost_s: csr_matrix, cost_t: csr_matrix, p_s: np.ndarray, p_t: np.ndarray,
                          idx2node_s: Dict, idx2node_t: Dict, ot_hyperpara: Dict) -> Tuple[List, List, List]:
    """
    Matching two graphs directly via calculate their Gromov-Wasserstein discrepancy.
    Args:
        cost_s: a (n_s, n_s) adjacency matrix of source graph
        cost_t: a (n_t, n_t) adjacency matrix of target graph
        p_s: a (n_s, 1) vector representing the distribution of source nodes
        p_t: a (n_t, 1) vector representing the distribution of target nodes
        idx2node_s: a dictionary {key: idx of cost_s's row, value: the name of source node}
        idx2node_t: a dictionary {key: idx of cost_s's row, value: the name of source node}
        ot_hyperpara: a dictionary of hyperparameters
    Returns:
        pairs_idx: a list of node index pairs
        pairs_name: a list of node name pairs
        pairs_confidence: a list of confidence of node pairs
    """
    trans, d_gw, p_s = gromov_wasserstein_discrepancy(cost_s, cost_t, p_s, p_t, ot_hyperpara)
    pairs_idx, pairs_name, pairs_confidence = node_pair_assignment(trans, p_s, p_t, idx2node_s, idx2node_t)
    return pairs_idx, pairs_name, pairs_confidence


def indrect_graph_matching(costs: Dict, probs: Dict, p_t: np.ndarray,
                           idx2nodes: Dict, ot_hyperpara: Dict, weights: Dict = None) -> Tuple[List, List, List]:
    """
    Matching two or more graphs indirectly via calculate their Gromov-Wasserstein barycenter.
    costs: a dictionary of graphs {key: graph idx,
                                       value: (n_s, n_s) adjacency matrix of source graph}
        probs: a dictionary of graphs {key: graph idx,
                                       value: (n_s, 1) the distribution of source nodes}
        p_t: (n_t, 1) the distribution of target nodes
        idx2nodes: a dictionary of graphs {key: graph idx,
                                           value: a dictionary {key: idx of row in cost,
                                                                value: name of node}}
        ot_hyperpara: a dictionary of hyperparameters
        weights: a dictionary of graph {key: graph idx,
                                       value: the weight of the graph}
    Returns:
        set_idx: a list of node index paired set
        set_name: a list of node name paired set
        set_confidence: a list of confidence set of node pairs
    """
    cost_t, trans, _ = gromov_wasserstein_barycenter(costs, probs, p_t, ot_hyperpara, weights)
    set_idx, set_name, set_confidence = node_set_assignment(trans, probs, idx2nodes)
    return set_idx, set_name, set_confidence


def recursive_direct_graph_matching(cost_s: csr_matrix, cost_t: csr_matrix,
                                    p_s: np.ndarray, p_t: np.ndarray,
                                    idx2node_s: Dict, idx2node_t: Dict,
                                    ot_hyperpara: Dict, weights: Dict = None, predefine_barycenter: bool = False,
                                    cluster_num: int = 2, partition_level: int = 3,
                                    max_node_num: int = 200) -> Tuple[List, List, List]:
    """
    recursive direct graph matching combining graph partition and indirect graph matching.
    1) apply "multi-graph partition" recursively to get a list of sub-graph sets
    2) apply "direct graph matching" to each sub-graph sets
    We require n_s >= n_t
    Args:
        cost_s: a (n_s, n_s) adjacency matrix of source graph
        cost_t: a (n_t, n_t) adjacency matrix of target graph
        p_s: a (n_s, 1) vector representing the distribution of source nodes
        p_t: a (n_t, 1) vector representing the distribution of target nodes
        idx2node_s: a dictionary {key: idx of cost_s's row, value: the name of source node}
        idx2node_t: a dictionary {key: idx of cost_s's row, value: the name of source node}
        ot_hyperpara: a dictionary of hyperparameters
        weights: a dictionary of graph {key: graph idx,
                                       value: the weight of the graph}
        predefine_barycenter: False: learn barycenter, True: use predefined barycenter
        cluster_num: the number of clusters when doing graph partition
        partition_level: the number of partition levels
        max_node_num: the maximum number of nodes in a sub-graph
    Returns:
        set_idx: a list of node index paired set
        set_name: a list of node name paired set
        set_confidence: a list of confidence set of node pairs
    """
    # apply "multi-graph partition" recursively to get a list of sub-graph sets
    costs = {0: cost_s, 1: cost_t}
    probs = {0: p_s, 1: p_t}
    idx2nodes = {0: idx2node_s, 1: idx2node_t}
    costs_all, probs_all, idx2nodes_all = recursive_multi_graph_partition(costs, probs, idx2nodes, ot_hyperpara,
                                                                          weights, predefine_barycenter, cluster_num,
                                                                          partition_level, max_node_num)
    # apply "indirect graph matching" to each sub-graph sets
    # set_idx = []
    set_name = []
    set_confidence = []
    for i in range(len(costs_all)):
        # print('Matching: sub-graph pair {}/{}, #source node={}, #target node={}'.format(
        #     i+1, len(costs_all), len(idx2nodes_all[i][0]), len(idx2nodes_all[i][1])))
        ot_hyperpara['outer_iteration'] = max([len(idx2nodes_all[i][0]), len(idx2nodes_all[i][1])])
        subset_idx, subset_name, subset_confidence = direct_graph_matching(costs_all[i][0], costs_all[i][1],
                                                                           probs_all[i][0], probs_all[i][1],
                                                                           idx2nodes_all[i][0], idx2nodes_all[i][1],
                                                                           ot_hyperpara)
        # set_idx += subset_idx
        set_name += subset_name
        set_confidence += subset_confidence

    node2idx_s = {}
    for key in idx2node_s.keys():
        node = idx2node_s[key]
        node2idx_s[node] = key
    node2idx_t = {}
    for key in idx2node_t.keys():
        node = idx2node_t[key]
        node2idx_t[node] = key
    set_idx = []
    for pair in set_name:
        idx_s = node2idx_s[pair[0]]
        idx_t = node2idx_t[pair[1]]
        set_idx.append([idx_s, idx_t])

    return set_idx, set_name, set_confidence


def recursive_indirect_graph_matching(costs: Dict, probs: Dict, idx2nodes: Dict, ot_hyperpara: Dict,
                                      weights: Dict = None, predefine_barycenter: bool = False,
                                      cluster_num: int = 2, partition_level: int = 3, max_node_num: int = 200
                                      ) -> Tuple[List, List, List]:
    """
    recursive indirect graph matching combining graph partition and indirect graph matching.
    1) apply "multi-graph partition" recursively to get a list of sub-graph sets
    2) apply "indirect graph matching" to each sub-graph sets
    Args:
        costs: a dictionary of graphs {key: graph idx,
                                       value: (n_s, n_s) adjacency matrix of source graph}
        probs: a dictionary of graphs {key: graph idx,
                                       value: (n_s, 1) the distribution of source nodes}
        idx2nodes: a dictionary of graphs {key: graph idx,
                                           value: a dictionary {key: idx of row in cost,
                                                                value: name of node}}
        ot_hyperpara: a dictionary of hyperparameters
        weights: a dictionary of graph {key: graph idx,
                                       value: the weight of the graph}
        predefine_barycenter: False: learn barycenter, True: use predefined barycenter
        cluster_num: the number of clusters when doing graph partition
        partition_level: the number of partition levels
        max_node_num: the maximum number of nodes in a sub-graph
    Returns:
        set_idx: a list of node index paired set
        set_name: a list of node name paired set
        set_confidence: a list of confidence set of node pairs
    """
    # apply "multi-graph partition" recursively to get a list of sub-graph sets
    costs_all, probs_all, idx2nodes_all = recursive_multi_graph_partition(costs, probs, idx2nodes, ot_hyperpara,
                                                                          weights, predefine_barycenter, cluster_num,
                                                                          partition_level, max_node_num)

    # apply "indirect graph matching" to each sub-graph sets
    # set_idx = []
    set_name = []
    set_confidence = []
    for i in range(len(costs_all)):
        num_node_min = np.inf
        num_node_max = 0
        for k in costs_all[i].keys():
            if num_node_min > costs_all[i][k].shape[0]:
                num_node_min = costs_all[i][k].shape[0]
            if num_node_max < costs_all[i][k].shape[0]:
                num_node_max = costs_all[i][k].shape[0]
        # print('Matching: sub-graphs {}/{}, the minimum #nodes = {}, the maximum #nodes = {}'.format(
        #     i+1, len(costs_all), num_node_min, num_node_max))
        p_t = estimate_target_distribution(probs_all[i], num_node_min)
        ot_hyperpara['outer_iteration'] = num_node_max
        subset_idx, subset_name, subset_confidence = indrect_graph_matching(
            costs_all[i], probs_all[i], p_t, idx2nodes_all[i], ot_hyperpara, weights)
        # set_idx += subset_idx
        set_name += subset_name
        set_confidence += subset_confidence

    node2idxes = {}
    for key in idx2nodes.keys():
        idx2node = idx2nodes[key]
        node2idx = {}
        for idx in idx2node.keys():
            node = idx2node[idx]
            node2idx[node] = idx
        node2idxes[key] = node2idx

    set_idx = []
    for pair in set_name:
        idx = []
        for key in node2idxes.keys():
            idx.append(node2idxes[key][pair[key]])
        set_idx.append(idx)

    return set_idx, set_name, set_confidence


def gromov_wasserstein_barycenter(costs: Dict, p_s: Dict, p_center: np.ndarray,
                                  ot_hyperpara: Dict, weights: Dict = None) -> Tuple[np.ndarray, Dict, List]:
    """
    Multi-graph matching based on one-step Gromov-Wasserstein barycenter learning.
    Args:
        costs: a dictionary, whose keys are graph ids and values are (n_s, n_s) cost matrices of different graphs
        p_s: a dictionary, whose keys are graph ids and values ara (n_s, 1) distributions of nodes of different graphs
        p_center: (n_c, 1) array, the distribution of barycenter's nodes
        ot_hyperpara: the dictionary of hyperparameters to train the Gromov-Wasserstein barycenter.
        weights: a dictionary, whose keys are graph ids and values are the weights of the graphs
    Returns:
        barycenter: (n_c, n_c) the cost matrix corresponding to the barycenter graph
        transports: a dictionary whose keys are graph ids and values are (n_s, n_c) optimal transports
        d_gw_sum: the sum of Gromov-Wasserstein discrepancy over iterations
    """
    # initialization
    num = len(costs)
    transports = {}
    for n in costs.keys():
        transports[n] = np.matmul(p_s[n], p_center.T)

    if weights is None:
        weights = {}
        for n in costs.keys():
            weights[n] = 1 / num

    # barycenter0 = np.random.rand(p_center.shape[0], p_center.shape[0])
    barycenter0 = csr_matrix(np.diag(p_center[:, 0]))

    d_gw_sum = []
    i = 0
    relative_error = np.inf
    while relative_error > ot_hyperpara['cost_bound'] and i < ot_hyperpara['max_iter']:
        # update optimal transport
        d_gw = {}
        for n in costs.keys():
            transports[n], d_gw[n], p_s[n] = gromov_wasserstein_discrepancy(costs[n], barycenter0,
                                                                            p_s[n], p_center,
                                                                            ot_hyperpara, transports[n])
        # averaging cost matrix
        barycenter = gromov_wasserstein_average(transports, costs, p_center, weights, ot_hyperpara['loss_type'])
        relative_error = np.sum(np.abs(barycenter - barycenter0)) / np.sum(np.abs(barycenter0))
        i += 1
        barycenter0 = barycenter
        d_gw_sum.append(d_gw)
    # print('barycenter iteration = {}'.format(i))
    return barycenter0, transports, d_gw_sum



def gromov_wasserstein_barycenter_simple(barycenter0, costs, bary_size=60, ot_hyperpara=None):
    """
    Multi-graph matching based on one-step Gromov-Wasserstein barycenter learning.
    Args:
        costs: a dictionary, whose keys are graph ids and values are (n_s, n_s) cost matrices of different graphs
        p_s: a dictionary, whose keys are graph ids and values ara (n_s, 1) distributions of nodes of different graphs
        p_center: (n_c, 1) array, the distribution of barycenter's nodes
        ot_hyperpara: the dictionary of hyperparameters to train the Gromov-Wasserstein barycenter.
        weights: a dictionary, whose keys are graph ids and values are the weights of the graphs
    Returns:
        barycenter: (n_c, n_c) the cost matrix corresponding to the barycenter graph
        transports: a dictionary whose keys are graph ids and values are (n_s, n_c) optimal transports
        d_gw_sum: the sum of Gromov-Wasserstein discrepancy over iterations
    """

    d_gw_sum = []
    i = 0
    relative_error = np.inf
    transports = {}
    p_s = {k: np.ones(shape=(costs[k].shape[0], 1)) for k in costs} # figure this out
    p_center = np.ones(shape=(bary_size, 1))
    for n in costs.keys():
        transports[n] = np.matmul(p_s[n], p_center.T)

    # costs should be all our adjacency matrices? or some inverse?
    weights = {k:1. for k in costs}



    while relative_error > ot_hyperpara['cost_bound'] and i < ot_hyperpara['max_iter']:
        # update optimal transport
        d_gw = {}
        for n in costs.keys():
            transports[n], d_gw[n], p_s[n] = gromov_wasserstein_discrepancy(costs[n], barycenter0,
                                                                            p_s[n], p_center,
                                                                            ot_hyperpara, transports[n])
        # averaging cost matrix
        barycenter = gromov_wasserstein_average(transports, costs, p_center, weights, ot_hyperpara['loss_type'])
        relative_error = np.sum(np.abs(barycenter - barycenter0)) / np.sum(np.abs(barycenter0))
        i += 1
        barycenter0 = barycenter
        d_gw_sum.append(d_gw)
    # print('barycenter iteration = {}'.format(i))
    return barycenter0, transports, d_gw_sum

'''
    cost_st = node_cost_st(cost_s, cost_t, p_s, p_t,
                           loss_type=ot_hyperpara['loss_type'], prior=ot_hyperpara['node_prior'])
    while relative_error > ot_hyperpara['iter_bound'] and t < ot_hyperpara['outer_iteration']:
        # update optimal transport via Sinkhorn iteration method
        cost = node_cost(cost_s, cost_t, trans0, cost_st, ot_hyperpara['loss_type'])
        if ot_hyperpara['ot_method'] == 'proximal':
            trans, a = sinkhorn_knopp_iteration(cost=cost,
                                                p_s=p_s,
                                                p_t=p_t,
                                                a=a,
                                                trans0=trans0,
                                                beta=ot_hyperpara['beta'],
                                                error_bound=ot_hyperpara['sk_bound'],
                                                max_iter=ot_hyperpara['inner_iteration'])
        else:
            trans, a = sinkhorn_knopp_iteration(cost=cost,
                                                p_s=p_s,
                                                p_t=p_t,
                                                a=a,
                                                trans0=None,
                                                beta=ot_hyperpara['beta'],
                                                error_bound=ot_hyperpara['sk_bound'],
                                                max_iter=ot_hyperpara['inner_iteration'])
        relative_error = np.sum(np.abs(trans - trans0)) / np.sum(np.abs(trans0))
        trans0 = trans
        t += 1

        # optionally, update source distribution
        if ot_hyperpara['update_p']:
            p_s, theta = update_distribution(a, p_s, theta,
                                             ot_hyperpara['beta'], ot_hyperpara['lr'], ot_hyperpara['alpha'])
    # print('proximal iteration = {}'.format(t))
    cost = node_cost(cost_s, cost_t, trans0, cost_st, ot_hyperpara['loss_type'])
    d_gw = (cost * trans0).sum()
    return trans0, d_gw, p_s



'''





def gromov_wasserstein_discrepancy(cost_s: csr_matrix, cost_t: csr_matrix,
                                   p_s: np.ndarray, p_t: np.ndarray,
                                   ot_hyperpara: Dict, trans0=None) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Calculate Gromov-Wasserstein discrepancy with optionally-updated source probability
    Args:
        cost_s: (n_s, n_s) np.ndarray of source cost matrix
        cost_t: (n_t, n_t) np.ndarray of target cost matrix
        p_s: (n_s, 1) np.ndarray, the predefined source distribution
        p_t: (n_t, 1) np.ndarray, the predefined target distribution
        ot_hyperpara: dictionary of hyperparameter
        trans0: optional (n_s, n_t) array, the initial transport
    Returns:
        trans0: (n_s, n_t) array, the optimal transport
        d_gw: a float representing Gromov-Wasserstein discrepancy
        p_s: (n_s, 1) array, the optimal source distribution
    """

    n_s = cost_s.shape[0]
    if ot_hyperpara['update_p']:
        theta = np.zeros((n_s, 1))
        p_s = softmax(theta)
    else:
        theta = np.zeros((n_s, 1))

    if trans0 is None:
        trans0 = np.matmul(p_s, p_t.T)

    a = np.ones((n_s, 1)) / n_s

    t = 0
    relative_error = np.inf
    # calculate invariant cost matrix
    cost_st = node_cost_st(cost_s, cost_t, p_s, p_t,
                           loss_type=ot_hyperpara['loss_type'], prior=ot_hyperpara['node_prior'])
    while relative_error > ot_hyperpara['iter_bound'] and t < ot_hyperpara['outer_iteration']:
        # update optimal transport via Sinkhorn iteration method
        cost = node_cost(cost_s, cost_t, trans0, cost_st, ot_hyperpara['loss_type'])
        if ot_hyperpara['ot_method'] == 'proximal':
            trans, a = sinkhorn_knopp_iteration(cost=cost,
                                                p_s=p_s,
                                                p_t=p_t,
                                                a=a,
                                                trans0=trans0,
                                                beta=ot_hyperpara['beta'],
                                                error_bound=ot_hyperpara['sk_bound'],
                                                max_iter=ot_hyperpara['inner_iteration'])
        else:
            trans, a = sinkhorn_knopp_iteration(cost=cost,
                                                p_s=p_s,
                                                p_t=p_t,
                                                a=a,
                                                trans0=None,
                                                beta=ot_hyperpara['beta'],
                                                error_bound=ot_hyperpara['sk_bound'],
                                                max_iter=ot_hyperpara['inner_iteration'])
        relative_error = np.sum(np.abs(trans - trans0)) / np.sum(np.abs(trans0))
        trans0 = trans
        t += 1

        # optionally, update source distribution
        if ot_hyperpara['update_p']:
            p_s, theta = update_distribution(a, p_s, theta,
                                             ot_hyperpara['beta'], ot_hyperpara['lr'], ot_hyperpara['alpha'])
    # print('proximal iteration = {}'.format(t))
    cost = node_cost(cost_s, cost_t, trans0, cost_st, ot_hyperpara['loss_type'])
    d_gw = (cost * trans0).sum()
    return trans0, d_gw, p_s



def node_cost_st(cost_s: csr_matrix, cost_t: csr_matrix,
                 p_s: np.ndarray, p_t: np.ndarray, loss_type: str = 'L2', prior: float = None) -> np.ndarray:
    """
    Calculate invariant cost between the nodes in different graphs based on learned optimal transport
    Args:
        cost_s: (n_s, n_s) array, the cost matrix of source graph
        cost_t: (n_t, n_t) array, the cost matrix of target graph
        p_s: (n_s, 1) array, the distribution of source nodes
        p_t: (n_t, 1) array, the distribution of target nodes
        loss_type: 'L2' the Euclidean loss type for Gromov-Wasserstein discrepancy
                   'KL' the KL-divergence loss type for Gromov-Wasserstein discrepancy
        prior: whether use node distribution similarity matrix as a prior
    Returns:
        cost_st: (n_s, n_t) array, the estimated invariant cost between the nodes in two graphs
    """
    n_s = cost_s.shape[0]
    n_t = cost_t.shape[0]
    if loss_type == 'L2':
        # f1(a) = a^2, f2(b) = b^2, h1(a) = a, h2(b) = 2b
        # cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
        # cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T

        # f1_st = np.repeat(np.matmul(cost_s ** 2, p_s), n_t, axis=1)
        # f2_st = np.repeat(np.matmul(p_t.T, (cost_t ** 2).T), n_s, axis=0)
        f1_st = np.repeat((cost_s ** 2) @ p_s, n_t, axis=1)
        f2_st = np.repeat(((cost_t ** 2) @ p_t).T, n_s, axis=0)
    else:
        # f1(a) = a*log(a) - a, f2(b) = b, h1(a) = a, h2(b) = log(b)
        # cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
        # cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T

        f1_st = np.repeat(np.matmul(cost_s * np.log(cost_s + 1e-15) - cost_s, p_s), n_t, axis=1)
        # f2_st = np.repeat(np.matmul(p_t.T, cost_t.T), n_s, axis=0)
        f2_st = np.repeat((cost_t @ p_t).T, n_s, axis=0)

    cost_st = f1_st + f2_st
    if prior is not None:
        cost_st += (prior * node_distribution_similarity(p_s, p_t))

    return cost_st


def node_cost(cost_s: csr_matrix, cost_t: csr_matrix, trans: np.ndarray,
              cost_st: np.ndarray, loss_type: str = 'L2') -> np.ndarray:
    """
    Calculate the cost between the nodes in different graphs based on learned optimal transport
    Args:
        cost_s: (n_s, n_s) array, the cost matrix of source graph
        cost_t: (n_t, n_t) array, the cost matrix of target graph
        trans: (n_s, n_t) array, the learned optimal transport between two graphs
        cost_st: (n_s, n_t) array, the estimated invariant cost between the nodes in two graphs
        loss_type: 'L2' the Euclidean loss type for Gromov-Wasserstein discrepancy
                   'KL' the KL-divergence loss type for Gromov-Wasserstein discrepancy
    Returns:
        cost: (n_s, n_t) array, the estimated cost between the nodes in two graphs
    """
    if loss_type == 'L2':
        # f1(a) = a^2, f2(b) = b^2, h1(a) = a, h2(b) = 2b
        # cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
        # cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T

        # cost = cost_st - 2 * np.matmul(np.matmul(cost_s, trans), cost_t.T)
        cost = cost_st - 2 * (cost_s @ trans @ cost_t.T)
    else:
        # f1(a) = a*log(a) - a, f2(b) = b, h1(a) = a, h2(b) = log(b)
        # cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
        # cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T

        # cost = cost_st - np.matmul(np.matmul(cost_s, trans), (np.log(cost_t + 1e-15)).T)
        cost = cost_st - np.matmul(cost_s @ trans, (np.log(cost_t + 1e-15)).T)
    return cost


def gromov_wasserstein_average(transports: Dict, costs: Dict,
                               p_center: np.ndarray, weights: Dict, loss_type: str) -> np.ndarray:
    """
    Averaging of cost matrix
    Args:
        transports: a dictionary, whose keys are graph ids and values are (n_s, n_c) np.ndarray of optimal transports
        costs: a dictionary, whose keys are graph ids and values are (n_s, n_s) np.ndarray of cost matrices
        p_center: (n_c, 1) np.ndarray of barycenter's distribution
        weights: a dictionary, whose keys are graph ids and values are float number of weight
        loss_type: 'L2' the Euclidean loss type for Gromov-Wasserstein discrepancy
                   'KL' the KL-divergence loss type for Gromov-Wasserstein discrepancy
    Returns:
        barycenter: (N, N) np.ndarray, the barycenter of cost matrix
    """
    barycenter = 0
    if loss_type == 'L2':
        for n in costs.keys():
            cost = costs[n]
            trans = transports[n]
            # barycenter += weights[n] * np.matmul(np.matmul(trans.T, cost), trans)
            barycenter += weights[n] * (trans.T @ (cost @ trans))
        barycenter /= np.matmul(p_center, p_center.T)
    else:
        for n in costs.keys():
            cost = costs[n]
            trans = transports[n]
            barycenter += weights[n] * np.matmul(np.matmul(trans.T, np.log(cost + 1e-15)), trans)
        barycenter /= np.matmul(p_center, p_center.T)
        barycenter = np.exp(barycenter)
    return barycenter


def sinkhorn_knopp_iteration(cost: np.ndarray, p_s: np.ndarray = None, p_t: np.ndarray = None,
                             a: np.ndarray = None, trans0: np.ndarray = None,
                             beta: float = 1e-1, error_bound: float = 1e-3,
                             max_iter: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sinkhorn-Knopp iteration algorithm
    When initial optimal transport "trans0" is not available, the function solves
        min_{trans in Pi(p_s, p_t)} <cost, trans> + beta * <log(trans), trans>
    When initial optimal transport "trans0" is given, the function solves:
        min_{trans in Pi(p_s, p_t)} <cost, trans> + beta * KL(trans || trans0)
    Args:
        cost: (n_s, n_t) array representing distance between nodes
        p_s: (n_s, 1) array representing the distribution of source nodes
        p_t: (n_t, 1) array representing the distribution of target nodes
        a: (n_s, 1) array representing the dual variable
        trans0: (n_s, n_t) initial array of optimal transport
        beta: the weight of entropic regularizer
        error_bound: the error bound to check convergence
        max_iter: the maximum number of iterations
    Returns:
        trans: optimal transport
        a: updated dual variable
    """
    if p_s is None:
        p_s = np.ones((cost.shape[0], 1)) / cost.shape[0]

    if p_t is None:
        p_t = np.ones((cost.shape[1], 1)) / cost.shape[1]

    if a is None:
        a = np.ones((cost.shape[0], 1)) / cost.shape[0]

    # cost /= np.max(cost)
    if trans0 is not None:
        kernel = np.exp(-cost / beta) * trans0
    else:
        kernel = np.exp(-cost / beta)

    relative_error = np.inf
    b = []
    i = 0
    # print(a)
    while relative_error > error_bound and i < max_iter:
        b = p_t / (np.matmul(kernel.T, a))
        a_new = p_s / np.matmul(kernel, b)
        relative_error = np.sum(np.abs(a_new - a)) / np.sum(np.abs(a))
        a = a_new
        i += 1
    trans = np.matmul(a, b.T) * kernel
    # print('sinkhorn iteration = {}'.format(i))
    return trans, a


def node_distribution_similarity(p_s: np.ndarray, p_t: np.ndarray, values: list=None) -> np.ndarray:
    """
    Calculate the node distribution similarity matrix
    Args:
        p_s: (n_s, 1) array representing the distribution of source node
        p_t: (n_t, 1) array representing the distribution of target node
    Returns:
        cost_st: (n_s, n_t) the cost matrix between node probability
    """
    # index_s = np.argsort(p_s[:, 0]) / p_s.shape[0]
    # index_s = np.reshape(index_s, p_s.shape)
    # index_t = np.argsort(p_t[:, 0]) / p_t.shape[0]
    # index_t = np.reshape(index_t, p_t.shape)
    # cost_st = (np.repeat(index_s, p_t.shape[0], axis=1) - np.repeat(index_t, p_s.shape[0], axis=1).T) ** 2\
    #             - 2 * index_s @ index_t.T
    if values is None:
        cost_st = (np.repeat(p_s, p_t.shape[0], axis=1) -
                   np.repeat(p_t, p_s.shape[0], axis=1).T) ** 2  # - 2 * p_s @ p_t.T
    else:
        cost_st = (np.repeat(values[0] * p_s, p_t.shape[0], axis=1) -
                   np.repeat(values[1] * p_t, p_s.shape[0], axis=1).T) ** 2  # - 2 * p_s @ p_t.T
    return cost_st


def softmax_grad(x: np.ndarray) -> np.ndarray:
    """
    The gradient of softmax function
    Args:
        x: (N, 1) or (N, ) array representing a distribution generated by softmax function
    Returns:
        grad_x: (N, N) array, the Jacobian matrix representing the gradient of softmax
    """
    s = x.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


def update_distribution(a: np.ndarray, p_s0: np.ndarray, theta0: np.ndarray,
                        beta: float, lr: float, weight: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update distribution via gradient descent
    Args:
        a: (n_s, 1) dual vector
        p_s0: (n_s, 1) current distribution
        theta0: (n_s, 1) current parameters of the distribution
        beta: the weight of first term
        lr: the learning rate
        weight: the weight of second term (regularizer)
    Returns:
        p_s: (n_s, 1) array of updated distribution
        theta: (n_s, 1) array of updated parameters
    """
    # update source distribution
    # grad_ps = beta * (np.log(a) - np.matmul((np.matmul(np.log(a), all1.transpose()) / kernel), all1))
    grad_ps = beta * np.log(a)
    if weight > 0:
        grad_ps -= (weight * (np.log(p_s0) + 1))
    grad_theta = np.matmul(softmax_grad(p_s0), grad_ps)
    # normalization
    grad_theta -= np.mean(grad_theta)
    grad_theta /= (1e-10 + np.sum(grad_theta ** 2) ** 0.5)
    theta = theta0 - lr * grad_theta
    p_s = softmax(theta)
    return p_s, theta
