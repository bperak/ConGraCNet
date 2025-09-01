import networkx as nx

def count_cycles_containing_edges(G):
    # nx.simple_cycles doesn't work with undirected graphs, use nx.cycle_basis
    cycles = nx.cycle_basis(G)
    print(cycles)
    edge_cycle_count = {frozenset(edge): 0 for edge in G.edges()}
    for cycle in cycles:
        # Create cycle edges considering undirected nature
        cycle_edges = [frozenset([cycle[i], cycle[(i + 1) % len(cycle)]]) for i in range(len(cycle))]
        for edge in cycle_edges:
            edge_cycle_count[edge] += 1
    return edge_cycle_count


def calculate_node_strengths(G):
    node_strength = {}
    for u, v, data in G.edges(data=True):
        weight = data.get('weight')  # Default weight to 1 if not specified
        if u not in node_strength:
            node_strength[u] = 0
        if v not in node_strength:
            node_strength[v] = 0
        node_strength[u] += weight
        node_strength[v] += weight
    
    return node_strength

def calculate_importance(G, edge_cycle_count, node_strength):
    importance_scores = {}
    for node in G.nodes():
        product = 0
        for neighbor in G.neighbors(node):
            edge = frozenset([node, neighbor])
            cycle_count = edge_cycle_count[edge]+1
            factor = node_strength[node] + node_strength[neighbor] - 2 * G[node][neighbor]['weight']
            w = G[node][neighbor]['weight']
            ratio = node_strength[node] / (node_strength[node] + node_strength[neighbor])
            partial_product = cycle_count * factor * w * ratio
            print((node, neighbor), cycle_count, node_strength[node], node_strength[neighbor], G[node][neighbor]['weight'], factor)
            product += partial_product

        # Directly use node strength for adjusting the product
        adjusted_product = product + node_strength[node]
        importance_scores[node] = adjusted_product

    # Normalize the scores
    total = sum(importance_scores.values())
    normalized_scores = {node: score / total * 100 for node, score in importance_scores.items()}
    return normalized_scores

def main():
    G = nx.Graph()
    weighted_edges = [
        (1, 2, 3), (1, 4, 3), (2, 1, 0.8), (4, 3, 1.4), (1, 5, 2), (1, 6, 2.5), (1, 7, 0.4),  (2, 20, 0.8), (2, 22, 0.7), (3, 1, 1.75), (3, 31, 1), (3, 40, 0.5),  (4, 41, 0.75), (4, 2, 0.4), (5, 52, 1), (5, 51, 0.8), (5, 50, 0.4), (6, 5, 1.5), (6, 20, 1.5),(6, 61, 0.5), (6, 60, 1.2), (7, 72, 0.4), (7, 70, 2), (21, 2, 0.8), (23, 2, 1), (32, 3, 0.4), (30, 3, 0.6), (40, 4, 1.5), (42, 4, 0.4), (52, 6, 0.4), (60, 2, 1.5), (71, 7, 0.8), (7, 73, 1), (73, 7, 1), (73, 732, 0.5), (731, 73, 0.5), (733, 73, 0.5)
    ]
    G.add_weighted_edges_from(weighted_edges)


    edge_cycle_count = count_cycles_containing_edges(G)
    node_strength = calculate_node_strengths(G)
    importance_scores = calculate_importance(G, edge_cycle_count, node_strength)

    for node, score in sorted(importance_scores.items(), key=lambda item: item[1], reverse=True):
        print(f"Node {node}: importance score = {score:.2f}%")

# Execute the main function
main()








