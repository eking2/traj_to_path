import pytraj as pt
import numpy as np
from typing import List, Tuple, Any, Dict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path
from tempfile import NamedTemporaryFile
import subprocess
import shlex


def load_traj(traj_path: str, parm_path: str, stride: int = 10) -> pt.Trajectory:

    """
    Read trajectory, autoimage and RMSD align to first frame.
    
    Parameters
    ----------
    traj_path : str
        Path to .nc file
    parm_path : str
        Path to .prmtop file
    stride : int (default: 10)
        Frame stride to subsample
        
    Returns
    -------
    traj : pt.Trajectory
        Autoimaged and RMSD aligned trajectory
    """

    # load
    traj = pt.load(traj_path, top=parm_path, stride=stride)

    # auto and align to first frame on alpha carbons
    traj = traj.autoimage()
    rmsd = pt.rmsd(traj, mask="@CA")

    return traj


def get_top_clf(n_clusters: np.array, scores: np.array, classifiers: Dict[str, KMeans]) -> Tuple[KMeans, int]:

    """
    Get top kmeans classifier based on silhouette score.

    Parameters
    ----------
    n_clusters : np.array
        Array of clusters tested
    scores : np.array
        Silhouette scores for each classifier
    classifiers : Dict
        Dict with keys number of clusters and values the fit KMeans classifier

    Returns
    -------
    clf : KMeans
        KMeans classifier fit with optimal number of clusters
    clusters : int
        Optimal number of clusters for kmeans
    """
    
    # index top score
    top_idx = np.argmax(scores)

    # select
    clusters = n_clusters[top_idx]
    clf = classifiers[str(clusters)]
    
    return clf, clusters


def silhouette_scores(traj: pt.Trajectory, min_clusters: int=2, max_clusters: int=10) -> Tuple[np.array, np.array, np.array, dict]:

    """
    KMeans on trajectory coordinates to lump conformations.
    
    Parameters
    ----------
    traj : pt.Trajectory
        MD trajectory
    min_clusters : int
        Min number of clusters for kmeans
    max_clusters : int
        Max number of clusters for kmeans

    Returns
    -------
    traj_pca : np.array
        Trajectory reduced to 2-dim by PCA, shape: (n_frames, 2)
    n_clusters : np.array
        Array of clusters tested
    scores : np.array
        Silhouette scores for each cluster
    classifiers : Dict[str, KMeans]
        Dict with keys number of clusters and values the fit KMeans classifier
    """
    
    classifiers = {}
    
    # alpha carbon coordinates
    traj = traj['@CA'].xyz.reshape(traj.shape[0], -1)

    # dim reduce
    pipe = make_pipeline(StandardScaler(), PCA(n_components=2))
    traj_pca = pipe.fit_transform(traj)
    
    # kmeans scan over range of cluster sizes
    n_clusters = np.arange(min_clusters, max_clusters + 1)
    scores = np.zeros(len(n_clusters))
    for i, cluster in enumerate(n_clusters):
        kmeans = KMeans(n_clusters=cluster)
        labels = kmeans.fit_predict(traj_pca)
        
        # score fit by silhouette
        sil = silhouette_score(traj_pca, labels)
        scores[i] = sil
        
        classifiers[str(cluster)] = kmeans
        
    return traj_pca, n_clusters, scores, classifiers


def get_representative_frame(traj: pt.Trajectory, traj_pca: np.array, clf: Any, name: str = 'rep') -> int:
     
    # get closest frame to most populated cluster
    labels = clf.predict(traj_pca)
    label_counts = np.bincount(labels)
    top_label = np.argmax(label_counts)
    
    dists = clf.transform(traj_pca)
    frame_idx = np.argmin(dists[:, top_label])
    
    pt.write_traj(f'{name}.pdb', traj['!:WAT,Na+,Cl-'], frame_indices=[frame_idx], overwrite=True)

    return frame_idx


def run_cpptraj(traj_path: str, parm_path: str, frame_idx: int) -> None:

    # get alpha carbon dist and corr mat

    template=f'''parm {parm_path}
trajin {traj_path}

reference {traj_path} {frame_idx}

autoimage
rms reference @CA
matrix dist @CA out distmat.dat
matrix correl @CA out corrmat.dat'''

    with NamedTemporaryFile() as cpptraj_in:
        cpptraj_in.write(template)
        cmd = f'cpptraj -i {cpptraj_in.name}'
        subprocess.run(shlex.split(cmd))