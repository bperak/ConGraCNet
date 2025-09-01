import networkx as nx

def main():
    G = nx.Graph()

    # Adding cycles
    nx.add_cycle(G, [1, 2, 4])
    nx.add_cycle(G, [1, 3, 4])
    nx.add_cycle(G, [1, 5, 6])
    nx.add_cycle(G, [3, 4, 40])
    nx.add_cycle(G, [5, 6, 52])

    # Adding weighted edges
    edges_tuple = [
        (1, 2, 3), (1, 3, 1.75), (1, 4, 3), (1, 5, 2), (1, 6, 2.5), (1, 7, 0.4),
        (2, 4, 0.4), (2, 20, 0.8), (2, 21, 2), (2, 22, 0.7), (2, 23, 1), (2, 60, 1.5),
        (3, 4, 1.4), (3, 40, 0.5), (3, 30, 0.6), (3, 31, 1), (3, 32, 0.4),
        (4, 40, 1.5), (4, 41, 0.75), (4, 42, 0.4),
        (5, 6, 1.5), (5, 50, 0.4), (5, 51, 0.8), (5, 52, 1),
        (6, 52, 0.4), (6, 60, 1.2), (6, 61, 0.5),
        (7, 70, 2), (7, 71, 0.8), (7, 72, 0.4)
    ]
    G.add_weighted_edges_from(edges_tuple, weight='weight')

    # Calculate and normalize the importance scores
    normalized_importance_scores = calculate_node_importance_and_normalize(G)

    # Print the normalized importance scores
    print("Normalized Importance Scores:")
    for node, score in normalized_importance_scores.items():
        print(f"Node {node}: {score:.2f}%")

def calculate_node_importance_and_normalize(G):
    # Calculate weighted degree for each node
    weighted_degree = {node: sum(data['weight'] for _, _, data in G.edges(node, data=True)) for node in G.nodes()}

    # Calculate importance scores for each node
    importance_scores = {}
    for node in G.nodes():
        node_weight = weighted_degree[node]
        importance = 0
        for neighbor in G.neighbors(node):
            edge_data = G.get_edge_data(node, neighbor)
            edge_weight = edge_data['weight']
            neighbor_weight = weighted_degree[neighbor]
            factor = (node_weight + neighbor_weight - 2 * edge_weight) * edge_weight
            importance += factor
        importance_scores[node] = importance + node_weight

    # Normalize the importance scores
    total_sum = sum(importance_scores.values())
    normalized_scores = {node: (score / total_sum) * 100 for node, score in importance_scores.items()}

    return normalized_scores

if __name__ == "__main__":
    main()
