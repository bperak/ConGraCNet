import networkx as nx
import numpy as np

def katz_bonacich_centrality(G, alpha=0.1, beta=1, tol=1e-6, max_iter=100):
    """
    Compute Katz-Bonacich centrality for nodes in a directed and weighted graph.

    Parameters:
    - G: NetworkX directed graph
    - alpha: Damping factor (default=0.1)
    - beta: Constant added to centrality scores (default=1)
    - tol: Tolerance for convergence (default=1e-6)
    - max_iter: Maximum number of iterations (default=100)

    Returns:
    - katz_cent: Dictionary of nodes with their Katz-Bonacich centrality scores
    """

    # Check if nodes have 'label' attribute, otherwise, use node identifiers as labels
    for node in G.nodes:
        if 'label' not in G.nodes[node]:
            G.nodes[node]['label'] = node

    # Initialize centrality scores
    katz_cent = {node: 0.0 for node in G.nodes}

    # Compute weighted adjacency matrix
    n = len(G.nodes)
    A = np.zeros((n, n))
    node_to_index = {node: i for i, node in enumerate(G.nodes)}

    for source, target, weight in G.edges(data='weight', default=1):
        A[node_to_index[source]][node_to_index[target]] = weight

    # Compute Katz-Bonacich centrality iteratively
    for _ in range(max_iter):
        prev_katz_cent = katz_cent.copy()

        for node in G.nodes:
            for neighbor in G.neighbors(node):
                katz_cent[node] += alpha * A[node_to_index[node]][node_to_index[neighbor]] * prev_katz_cent[neighbor]

        # Add the beta constant to each node's centrality score
        for node in G.nodes:
            katz_cent[node] += beta

        # Normalize centrality scores
        norm_factor = max(katz_cent.values())
        katz_cent = {node: centrality / norm_factor for node, centrality in katz_cent.items()}

        # Check for convergence
        if np.allclose(list(prev_katz_cent.values()), list(katz_cent.values()), atol=tol):
            break

    return katz_cent

# Create the directed and weighted graph
G = nx.DiGraph()
edges = [(1, 2, {'weight': 3}), (1, 4, {'weight': 3}), (1, 5, {'weight': 2}), (1, 6, {'weight': 2.5}), (1, 7, {'weight': 0.4}),
         (2, 1, {'weight': 0.8}), (2, 20, {'weight': 0.8}), (2, 22, {'weight': 0.7}), (3, 1, {'weight': 1.75}), (3, 31, {'weight': 1}),
         (3, 40, {'weight': 0.5}), (4, 3, {'weight': 1.4}), (4, 41, {'weight': 0.75}), (4, 2, {'weight': 0.4}), (5, 52, {'weight': 1}),
         (5, 51, {'weight': 0.8}), (5, 50, {'weight': 0.4}), (6, 5, {'weight': 1.5}), (6, 20, {'weight': 1.5}), (6, 61, {'weight': 0.5}),
         (6, 60, {'weight': 1.2}), (7, 72, {'weight': 0.4}), (7, 70, {'weight': 2}), (21, 2, {'weight': 0.8}), (23, 2, {'weight': 1}),
         (32, 3, {'weight': 0.4}), (30, 3, {'weight': 0.6}), (40, 4, {'weight': 1.5}), (42, 4, {'weight': 0.4}), (52, 6, {'weight': 0.4}),
         (60, 2, {'weight': 1.5}), (71, 7, {'weight': 0.8}), (7, 73, {'weight': 1}), (73, 7, {'weight': 1}), (73, 732, {'weight': 0.5}),
         (731, 73, {'weight': 0.5}), (733, 73, {'weight': 0.5})]

G.add_edges_from(edges)

# Compute Katz-Bonacich centrality
bonacich_cent = katz_bonacich_centrality(G)

# Print centrality scores
print("Katz-Bonacich centrality scores:")
for node, centrality in bonacich_cent.items():
    print(f"Node {node}: {centrality}")
