import networkx as nx

# Initialize a directed graph
G = nx.DiGraph()

# Define weighted edges
weighted_edges = [
    (1, 2, 3), (1, 4, 3), (2, 1, 0.8), (4, 3, 1.4), (1, 5, 2), (1, 6, 2.5), (1, 7, 0.4),
    (2, 20, 0.8), (2, 22, 0.7), (3, 1, 1.75), (3, 31, 1), (3, 40, 0.5), (4, 41, 0.75),
    (4, 2, 0.4), (5, 52, 1), (5, 51, 0.8), (5, 50, 0.4), (6, 5, 1.5), (6, 20, 1.5),
    (6, 61, 0.5), (6, 60, 1.2), (7, 72, 0.4), (7, 70, 2), (21, 2, 0.8), (23, 2, 1),
    (32, 3, 0.4), (30, 3, 0.6), (40, 4, 1.5), (42, 4, 0.4), (52, 6, 0.4), (60, 2, 1.5),
    (71, 7, 0.8), (7, 73, 1), (73, 7, 1), (73, 732, 0.5), (731, 73, 0.5), (733, 73, 0.5)
]

# Add weighted edges to the graph
G.add_weighted_edges_from(weighted_edges)

# Compute the Katz centrality for the directed graph, considering the edge weights
centrality = nx.katz_centrality(G, alpha=0.1, beta=1.0, max_iter=1000, tol=1e-06, nstart=None, normalized=True, weight='weight')

# Print the Katz centrality for each node
for node, centrality_score in centrality.items():
    print(f"Node {node}: {centrality_score}")
