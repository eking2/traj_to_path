import numpy as np
import pandas as pd
import networkx as nx
from tqdm.auto import tqdm
from networkx.classes.graph import Graph
from typing import Dict, Tuple, List

"""
inputs
------
1. distance matrix
2. correlation matrix

steps
-----
1. read matrices
2. select pairs of residues within distance cutoff and use as edges
3. define protein graph
    - nodes are residues
    - edges are between residues within distance cutoff
    - weights are abs(-log10(corr))
4. get all shortest paths
    - from res i to i+2 up to last residue
5. make graph of shortest paths
    - save all edges observed (sequential nodes)
    - count frequency of all edges
6. normalize edge counts from shortest paths
    - divide by max
    - use as weight
7. analysis
    - average path length
    - betweenness
    - communities
        - pymol fig
"""


def mats_to_edges(
    dist_mat: np.ndarray, corr_mat: np.ndarray, cutoff: float = 6
) -> Tuple[pd.DataFrame, Graph]:

    """
    Select nodes based on distance cutoff and weight edges by correlation.

    Parameters
    ----------
    dist_mat : np.array
        C-alpha distance matrix (n_atoms x n_atoms)
    corr_mat : np.array
        C-alpha correlation matrix (n_atoms x n_atoms)
    cutoff : float (default: 6)
        Pairwise distance cutoff to select edges

    Returns
    -------
    graph_df : pd.DataFrame
        Edge list dataframe
    graph : Graph
        Graph with residue nodes and edges weighted by correlation
    """

    assert (
        dist_mat.shape == corr_mat.shape
    ), f"Shape mismatch: dist_mat {dist_mat.shape} != corr_mat {corr_mat.shape}"

    # set lower triangle values above cutoff
    # skip mirror and self
    tril_idx = np.tril_indices(dist_mat.shape[0])
    dist_mat[tril_idx] = cutoff + 1

    # edges where distance below cutoff
    edges = np.argwhere(dist_mat <= cutoff)

    # save source node, target node, weight
    edge_list = []

    for edge in edges:

        # high correlation or anticorrelation = smaller weight
        corr = np.abs(corr_mat[edge[0], edge[1]]) + 1e-9
        weight = np.abs(np.log10(corr))

        # to residue 1-index
        source = edge[0] + 1
        target = edge[1] + 1

        edge_list.append([source, target, weight])

    graph_df = pd.DataFrame(edge_list, columns=["source", "target", "weight"])
    graph = nx.convert_matrix.from_pandas_edgelist(
        graph_df, source="source", target="target", edge_attr="weight"
    )

    return graph_df, graph


def get_all_sp(graph: Graph, sep: int = 2, method: str = "dijkstra") -> List[List[int]]:

    """
    Calculate the shortest path between all pairs of separated residues.

    Parameters
    ----------
    graph : Graph
        NetworkX graph of trajectory
    sep : int (default: 2)
        Minimum sequence gap between start and end residue in graph
    method : str (default: 'dijkstra')
        Method to compute shortest path

    Returns
    -------
    paths : List[List[int]]
        List of shortest paths
    """

    last_node = max(set(graph.nodes))
    nodes = np.arange(1, last_node + 1)

    # expanding window from each start node
    paths = []
    for start_node in tqdm(nodes):

        end_node = start_node + sep
        while end_node <= last_node:
            path = nx.shortest_path(
                graph, start_node, end_node, weight="weight", method=method
            )
            paths.append(path)

            end_node += 1

    return paths


def sp_to_edges(paths: List[List[int]]) -> Tuple[List[List[int]], np.ndarray]:

    """
    Convert list of paths to edge array counting frequencies.

    Parameters
    ----------
    paths : List[List[int]]
        List of shortest paths

    Returns
    -------
    edge_list : List[List[int]]
        Edge list for shortest path graph
    edge_arr : np.ndarray
        Edge frequencies in shortest path graph
    """

    last_node = max([max(path) for path in paths])
    edge_list = []
    edge_arr = np.zeros((last_node, last_node), dtype=int)

    for path in paths:
        for i in range(len(path) - 1):

            # 1-index
            start_res = path[i]
            end_res = path[i + 1]

            # 0-index
            start_0 = start_res - 1
            end_0 = end_res - 1

            # only save unique edges
            count_a = edge_arr[start_0, end_0]
            count_b = edge_arr[end_0, start_0]

            if (count_a == 0) and (count_b == 0):
                edge_list.append([start_res, end_res])

            # update counts, keep symmetric
            # will not always be increasing order
            edge_arr[start_0, end_0] = count_a + 1
            edge_arr[end_0, start_0] = count_b + 1

    return edge_list, edge_arr


def sp_edges_to_graph(
    edge_list: List[List[int]], edge_arr: np.ndarray, cutoff: float = 0.25
) -> Tuple[pd.DataFrame, Graph]:

    """
    Make graph from central edges based on weight cutoff.

    Parameters
    ----------
    edge_list : List[List[int]]
        Edge list for shortest path graph
    edge_arr : np.ndarray
        Edge frequencies in shortest path graph
    cutoff : float (default: 0.3)
        Weight cutoff to keep highly connected edges

    Returns
    -------
    sel_df : pd.DataFrame
        Edge list dataframe
    sel_graph : Graph
        Graph with residue nodes and edges weighted by frequency in shortest paths
    """

    sel_edges = []

    # normalize frequencies
    max_freq = edge_arr.max()
    weighted_arr = edge_arr / max_freq

    # save edges with weight above cutoff
    for edge in edge_list:

        # to 0-index
        node_start = edge[0] - 1
        node_end = edge[1] - 1

        weight = weighted_arr[node_start, node_end]

        if weight >= cutoff:
            sel_edges.append([edge[0], edge[1], weight])

    sel_df = pd.DataFrame(sel_edges, columns=["source", "target", "weight"])
    sel_graph = nx.convert_matrix.from_pandas_edgelist(
        sel_df, source="source", target="target", edge_attr="weight"
    )

    return sel_df, sel_graph
