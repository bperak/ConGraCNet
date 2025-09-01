import networkx as nx

def count_cycles_containing_edges(G):
    # For undirected graphs, cycle basis can be directly used.
    cycles = nx.cycle_basis(G)
    edge_cycle_count = {}
    for cycle in cycles:
        for i in range(len(cycle)):
            edge = frozenset({cycle[i], cycle[(i + 1) % len(cycle)]})
            edge_cycle_count[edge] = edge_cycle_count.get(edge, 0) + 1
    return edge_cycle_count

def calculate_node_strengths(G):
    # In undirected graphs, there's no distinction between in-strength and out-strength.
    node_strength = {}
    for node in G.nodes:
        node_strength[node] = sum(G[node][neighbor]['weight'] for neighbor in G.neighbors(node))
    return node_strength

def calculate_importance(G, edge_cycle_count, node_strength):
    importance_scores = {}
    for node in G.nodes():
        product = 0
        for neighbor in G.neighbors(node):
            edge = frozenset({node, neighbor})
            cycle_count = edge_cycle_count.get(edge, 0)+1
            factor = node_strength[node] + node_strength[neighbor] - 2 * G[node][neighbor]['weight']
            w = G[node][neighbor]['weight']
            omjer = node_strength[node] / (node_strength[node] + node_strength[neighbor])
            partial_umnozak = (cycle_count + 1) * factor * w * omjer
            print((node, neighbor), cycle_count)
            product += partial_umnozak

        # Strength sum is simply the node strength since it's undirected.
        adjusted_umnozak = product + node_strength[node]
        importance_scores[node] = adjusted_umnozak

    # Normalize the scores
    total = sum(importance_scores.values())
    normalized_scores = {node: score / total * 100 for node, score in importance_scores.items()}
    return normalized_scores

def main():
    G = nx.Graph()
    weighted_edges = [
        (1, 2, 3), 
        (1, 4, 3), 
        (1, 5, 2), 
        (1, 6, 2.5), 
        (1, 7, 0.4), 
        (2, 1, 0.8), 
        (2, 20, 0.8),
        (2, 22, 0.7), 
        (3, 1, 1.75), 
        (3, 31, 1), 
        (3, 40, 0.5), 
        (4, 3, 1.4), 
        (4, 41, 0.75),
        (4, 2, 0.4), 
        (5, 52, 1), 
        (5, 51, 0.8), 
        (5, 50, 0.4), 
        (6, 5, 1.5), 
        (6, 20, 1.5),
        (6, 61, 0.5), 
        (6, 60, 1.2), 
        (7, 72, 0.4), 
        (7, 70, 2), 
        (21, 2, 0.8), 
        (23, 2, 1),
        (32, 3, 0.4), 
        (30, 3, 0.6), 
        (40, 4, 1.5), 
        (42, 4, 0.4), 
        (52, 6, 0.4), 
        (60, 2, 1.5),
        (71, 7, 0.8), 
        (7, 73, 1), 
        (73, 7, 1), 
        (73, 732, 0.5), 
        (731, 73, 0.5), 
        (733, 73, 0.5)
    ]
    G.add_weighted_edges_from(weighted_edges)


    edge_cycle_count = count_cycles_containing_edges(G)
    node_strength = calculate_node_strengths(G)
    importance_scores = calculate_importance(G, edge_cycle_count, node_strength)

    for node, score in sorted(importance_scores.items(), key=lambda item: item[1], reverse=True):
        print(f"Node {node} importance score: {score:.2f}%")

# Example usage with your specific edges
main()





