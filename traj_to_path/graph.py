'''
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
'''