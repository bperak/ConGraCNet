import networkx as nx

def find_cycles_and_update_edge_counts(G):
    # Use cycle_basis for undirected graphs
    cycles = nx.cycle_basis(G)
    edge_cycle_count = {}
    node_cycles = {node: [] for node in G.nodes()}

    for cycle in cycles:
        cycle_edges = [frozenset({cycle[i], cycle[(i + 1) % len(cycle)]}) for i in range(len(cycle))]
        for edge in cycle_edges:
            edge_cycle_count[edge] = edge_cycle_count.get(edge, 0) + 1
        # Add the actual cycle to the nodes involved
        for node in cycle:
            node_cycles[node].append(cycle)

    return edge_cycle_count, node_cycles

def calculate_node_strength(G):
    node_strength = {}
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1)  # Default weight to 1 if not specified
        if u not in node_strength:
            node_strength[u] = 0
        if v not in node_strength:
            node_strength[v] = 0
        node_strength[u] += weight
        node_strength[v] += weight
    return node_strength

def calculate_importance(G, edge_cycle_count, node_strength, node_cycles):
    importance_scores = {}
    for node in G.nodes():
        product = 0
        for neighbor in G.neighbors(node):
            edge = frozenset({node, neighbor})
            cycle_count = edge_cycle_count.get(edge, 0) + 1  # Add 1 to include the edge itself as a minimal cycle
            factor = node_strength[node] + node_strength[neighbor] - 2 * G[node][neighbor].get('weight', 1)
            omjer = node_strength[node] / (node_strength[node] + node_strength[neighbor])
            product += cycle_count * factor * omjer
        adjusted_product = product + node_strength[node]
        importance_scores[node] = adjusted_product

    # Normalize the scores
    total = sum(importance_scores.values())
    normalized_scores = {node: (score / total * 100) for node, score in importance_scores.items()}
    return normalized_scores, node_cycles

def main():
    G = nx.DiGraph()
    weighted_edges = [
        (1, 2, 3), 
        (1, 4, 2.9), (4, 1, 0.1),
        (1, 5, 1.9), (5, 1, 0.1), 
        (1, 6, 2.4),  (6, 1, 0.1),
        (1, 7, 0.3), (7, 1, 0.1),
        (2, 1, 0.8),
        (2, 20, 0.7), (20, 2, 0.1), 
        (2, 22, 0.6), (22, 2, 0.1), 
        (3, 1, 1.65), (1, 3, 0.1),
        (3, 31, 0.9), (31, 1, 0.1),
        (3, 40, 0.4), (40, 3, 0.1),
        (4, 3, 1.3), (3, 4, 0.1), 
        (4, 41, 0.65), (41, 4, 0.1),
        (4, 2, 0.3), (2, 4, 0.1), 
        (5, 52, 0.9), (52, 5, 0.1), 
        (5, 51, 0.7), (51, 5, 0.1), 
        (5, 50, 0.3),  (50, 5, 0.1),
        (6, 5, 1.4), (5, 6, 0.1), 
        (6, 20, 1.4), (20, 6, 0.1), 
        (6, 61, 0.4), (61, 6, 0.1), 
        (6, 60, 1.1), (60, 6, 0.1),
        (7, 72, 0.3), (72, 7, 0.1), 
        (7, 70, 1.9), (70, 7, 0.1), 
        (21, 2, 0.7), (2, 21, 0.1),
        (23, 2, 0.9), (2, 23, 0.1),
        (32, 3, 0.3), (3, 32, 0.1), 
        (30, 3, 0.5), (3, 30, 0.1), 
        (40, 4, 1.4), (4, 40, 0.1), 
        (42, 4, 0.3), (4, 42, 0.1), 
        (52, 6, 0.4), (6, 52, 0.1), 
        (60, 2, 1.4), (2, 60, 0.1), 
        (71, 7, 0.7), (7, 71, 0.1),
        (7, 73, 1), 
        (73, 7, 1), 
        (73, 732, 0.4), (732, 73, 0.1), 
        (731, 73, 0.4), (73, 731, 0.1),
        (733, 73, 0.4), (73, 733, 0.1)
    ]
    G.add_weighted_edges_from(weighted_edges)

    edge_cycle_count, node_cycles = find_cycles_and_update_edge_counts(G)
    node_strength = calculate_node_strength(G)
    importance_scores, node_cycles = calculate_importance(G, edge_cycle_count, node_strength, node_cycles)

    for node, score in sorted(importance_scores.items(), key=lambda item: item[1], reverse=True):
        print(f"Node {node}: Importance score = {score:.2f}%")
        print(f"Cycles containing node {node}: {node_cycles[node]}")

main()















