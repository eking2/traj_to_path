from traj_to_path import preprocess, graph, utils
from networkx.classes.graph import Graph


def get_traj_feats(
    traj_path: str,
    parm_path: str,
    stride: int = 10,
    min_clusters: int = 2,
    max_clusters: int = 10,
    name: str = "rep",
) -> None:

    """
    Parse trajectory to save alpha carbon distance matrix, correlations, and
    representative frame.

    Parameters
    ----------
    traj_path : str
        Path to .nc file
    parm_path :str
        Path to .prmtop file
    stride : int (default: 10)
        Frame stride to subsample
    min_clusters : int (default: 2)
        Min number of clusters for kmeans
    max_clusters : int (default: 10)
        Max number of clusters for kmeans
    name : str (default: 'rep')
        Output pdb filename
    """

    # load traj
    traj = preprocess.load_traj(traj_path, parm_path)

    # kmeans cluster
    traj_pca, n_clusters, scores, classifiers = preprocess.silhouette_scores(
        traj, min_clusters, max_clusters
    )

    # get optimal number of clusters
    clf, clusters = preprocess.get_top_clf(n_clusters, scores, classifiers)

    # save representative frame as ref
    frame_idx = preprocess.get_representative_frame(traj, traj_pca, clf)

    # cpptraj to get c-alpha distances and correlations
    preprocess.run_cpptraj(traj_path, parm_path, frame_idx)


def get_sp_graph(
    dist_path: str,
    corr_path: str,
    dist_cutoff: float = 6.0,
    res_sep: int = 2,
    sp_method: str = "dijkstra",
    sp_cutoff: float = 0.25,
) -> Graph:

    """
    Make shortest path graph from trajectory.

    Parameters
    ----------
    dist_path : str
        Path to distance matrix
    corr_path : str
        Path to correlation matrix
    dist_cutoff : float
        Pairwise distance cutoff to select edges
    res_sep : int
        Minimum sequence gap between start and end residue in graph
    sp_method : str
        Method to compute shortest path
    sp_cutoff : float
        Weight cutoff to keep highly connected edges

    Returns
    -------
    G : Graph
        Shortest path graph
    """

    # input dist and corr matrices
    dist_mat = utils.parse_mat(dist_path)
    corr_mat = utils.parse_mat(corr_path)

    # to traj graph
    traj_graph_df, traj_graph = graph.mats_to_edges(dist_mat, corr_mat, dist_cutoff)

    # get shortest paths from expanding window on all residues
    sp_list = graph.get_all_sp(traj_graph, res_sep, sp_method)

    # count edges in paths
    sp_edge_list, sp_edge_arr = graph.sp_to_edges(sp_list)

    # make sp graph
    df_G, G = graph.sp_edges_to_graph(sp_edge_list, sp_edge_arr, sp_cutoff)

    return G
