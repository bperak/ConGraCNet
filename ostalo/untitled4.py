import numpy as np
import networkx as nx
import pandas as pd

def katz_bonacich_centrality(adjacency_matrix, alpha=0.1, beta=1, max_iterations=100, tol=1e-6):
    n = len(adjacency_matrix)
    x = np.ones(n) * beta
    A = adjacency_matrix
    I = np.eye(n)
    convergence = False

    for _ in range(max_iterations):
        x_new = alpha * np.dot(A, x) + beta
        if np.linalg.norm(x_new - x) < tol:
            convergence = True
            break
        x = x_new

    if not convergence:
        print("Katz-Bonacich centrality did not converge.")

    return x

# Define the graph
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

# Get adjacency matrix
adjacency_matrix = nx.to_numpy_array(G, weight='weight')

# Compute Katz-Bonacich centrality
centrality_scores = katz_bonacich_centrality(adjacency_matrix)

def centrality_percentiles(scores):
    percentiles = []
    for score in scores:
        percentile = np.percentile(scores, np.sum(scores <= score) / len(scores) * 100)
        percentiles.append(percentile)
    return percentiles

# Assuming centrality_scores contains your Katz-Bonacich centrality scores


# Create DataFrame
df = pd.DataFrame({
    'Node': list(G.nodes()),
    'Katz-Bonacich Centrality': centrality_scores
})

print(df)

# Assuming 'df' is your DataFrame containing node labels and Katz-Bonacich centrality scores
centrality_sum = df['Katz-Bonacich Centrality'].sum()
print("Sum of Katz-Bonacich centrality scores:", centrality_sum)

# Assuming 'df' is your DataFrame containing node labels and Katz-Bonacich centrality scores
centrality_sum = df['Katz-Bonacich Centrality'].sum()

# Normalize centrality scores
df['Normalized Centrality'] = (df['Katz-Bonacich Centrality'] / centrality_sum) * 100

print(df)
