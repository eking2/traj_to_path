import numpy as np
import networkx as nx
from networkx.classes.graph import Graph
import matplotlib.pyplot as plt
from typing import Tuple, List
from pathlib import Path


def random_color() -> Tuple[int, int, int]:

    """Random RGB color."""
    color = list(np.random.choice(256, size=3))
    return color


def get_middle(lst: List[int]) -> int:

    """Get middle value from sorted list."""

    middle = len(lst) // 2
    return sorted(list(lst))[middle]


def plot_mat(mat: np.ndarray, mat_type: str = "dist", **kwargs) -> None:

    """Plot matrix heatmap.
    
    Args:
        mat (np.ndarray): Square array with alpha-carbon distance or correlation data.
        mat_type (str): Matrix type, either 'dist' or 'corr'. Default 'dist'
    """

    assert mat_type in ["dist", "corr"], f"invalid mat_type: {mat_type}"
    assert mat.shape[0] == mat.shape[1], f"must be square, invalid shape: {mat.shape}"

    if mat_type == "dist":
        cmap = "viridis"
        cbar_label = "Distance ($\AA$)"
        title = r"C$\alpha$ Distance Matrix"
    elif mat_type == "corr":
        cmap = "RdBu"
        cbar_label = r"Correlation ($\rho$)"
        title = r"C$\alpha$ Correlation Matrix"

    # hide upper tri
    mask = np.triu_indices(mat.shape[0], k=0)
    mat[mask] = np.nan

    fig = plt.figure(figsize=(7, 5))
    p = plt.imshow(mat, cmap=cmap, **kwargs)
    cbar = plt.colorbar(p)
    cbar.set_label(label=cbar_label, size=13)
    plt.title(title, size=14)

    plt.xlabel("Residue $i$", size=13)
    plt.ylabel("Residue $j$", size=13)
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def plot_graph(
    graph: Graph, scale: float = 20_000, layout: str = "spring", **kwargs
) -> None:

    """Plot graph layout.
    
    Args:
        graph (Graph): Input NetworkX graph.
        scale (float): Factor to scale node sizes. Default 20_000
        layout (str): Type of graph layout to plot. Default 'spring'
    """

    assert layout in [
        "spring",
        "kamada_kawai",
        "shell",
        "spectral",
    ], f"invalid layout: {layout}"

    # size nodes by degree centrality
    centrality = nx.betweenness_centrality(graph, endpoints=True)
    node_size = [c * scale for c in centrality.values()]

    # color by community label
    community = nx.community.label_propagation_communities(graph)
    com_idx = {n: i for i, com in enumerate(community) for n in com}
    node_color = [com_idx[n] for n in graph]

    # set node positions
    layout_dict = {
        "spring": nx.spring_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
        "shell": nx.shell_layout,
        "spectral": nx.spectral_layout,
    }
    pos = layout_dict[layout](graph)

    fig = plt.figure(figsize=(9, 8))
    nx.draw_networkx(
        graph, pos=pos, node_size=node_size, node_color=node_color, alpha=0.9, **kwargs
    )

    # annotate
    clusters = len(set(com_idx.values()))
    ax = plt.gca()
    ax.text(0.8, 0.05, f"{clusters} clusters", transform=ax.transAxes, size=14)


def pymol_com(graph: Graph, name: str = "traj_com", labels: bool = False) -> None:

    """Color pymol structure by community label.
    
    Args:
        graph (Graph): Trajectory graph weighted by correlation.
        name (str): Output filename. Default 'traj_com'
        labels (bool): Label clusters or leave blank. Default False
    """

    communities = list(nx.community.label_propagation_communities(graph))

    # write pml to overlay
    template = []
    for i, com in enumerate(communities):
        sel = "+".join(map(str, com))
        label = f"com_{i}"

        # color community
        col = random_color()
        sel_line = f"sel {label}, resi {sel}\nset_color {label}_c, {col}\ncolor {label}_c, {label}"
        template.append(sel_line)

        # label middle residue
        if labels:
            mid = get_middle(com)
            label_line = f'label name ca and resi {mid}, "{i}"'
            template.append(label_line)

    template.append("desel")
    out = "\n".join(template)
    Path(f"{name}.pml").write_text(out)
