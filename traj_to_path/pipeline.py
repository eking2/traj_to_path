from . import preprocess


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
