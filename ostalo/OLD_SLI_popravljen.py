import networkx as nx
import itertools

def calculate_node_importance_and_normalize(G):
    cycle_basis = nx.cycle_basis(G.to_undirected())

    # Initialize dict1 with counts for both directions of each edge
    dict1 = {}
    for edge in G.edges():
        dict1[(edge[0], edge[1])] = 0
        dict1[(edge[1], edge[0])] = 0  # Add reverse direction for undirected edge

    # Count cycles each edge is part of
    for cycle in cycle_basis:
        for i in range(len(cycle)):
            edge = (cycle[i], cycle[(i + 1) % len(cycle)])
            dict1[edge] += 1
            dict1[(edge[1], edge[0])] += 1  # Increment for reverse direction

    importance_scores = {}
    for node in G.nodes():
        sum_importance = 0
        weighted_degree_node = sum(data['weight'] for _, _, data in G.edges(node, data=True))
        for neighbor in G.neighbors(node):
            edge_weight = G[node][neighbor]['weight']
            weighted_degree_neighbor = sum(data['weight'] for _, _, data in G.edges(neighbor, data=True))
            u = (weighted_degree_node + weighted_degree_neighbor - 2 * edge_weight)
            lam = dict1[(node, neighbor)]+1
            z = edge_weight * weighted_degree_node / (weighted_degree_node + weighted_degree_neighbor)
            sum_importance += u * lam * z
            #print()
        total_importance = sum_importance + weighted_degree_node
        importance_scores[node] = total_importance

    # Normalize the importance scores
    total_sum = sum(importance_scores.values())
    normalized_scores = {node: score / total_sum * 100 for node, score in importance_scores.items()}
    return normalized_scores

G = nx.Graph()

#nx.add_cycle(G, [10,20,30])
#nx.add_cycle(G, [3, 4, 26])
nx.add_cycle(G, [1, 2, 4])
nx.add_cycle(G, [1, 3, 4])
nx.add_cycle(G, [1, 5, 6])
nx.add_cycle(G, [3, 4, 40])
#nx.add_cycle(G, [1, 2, 20, 6])
nx.add_cycle(G, [5, 6, 52])
#nx.add_cycle(G, [2, 60, 6, 20])
#nx.add_cycle(G, [1, 3, 8, 6])

#G.add_edge(1, 10, weight=0.4)
G.add_edge(1, 2, weight=3)
G.add_edge(1, 3, weight=1.75)
G.add_edge(1, 4, weight=3)
#G.add_edge(4, 10, weight=0.4)
G.add_edge(1, 5, weight=2)
G.add_edge(1, 6, weight=2.5)
G.add_edge(1, 7, weight=0.4)
G.add_edge(2, 4, weight=0.4)
G.add_edge(2, 20, weight=0.8)
G.add_edge(2, 21, weight=2)
G.add_edge(2, 22, weight=0.7)
G.add_edge(2, 23, weight=1)
G.add_edge(2, 60, weight=1.5)
#G.add_edge(3, 8, weight=0.4)
G.add_edge(3, 4, weight=0.4)
G.add_edge(3, 40, weight=0.5)
#G.add_edge(4, 2, weight=0.4)
G.add_edge(3, 30, weight=0.6)
G.add_edge(3, 31, weight=1)
G.add_edge(3, 32, weight=0.4)
G.add_edge(4, 40, weight=1.5)
G.add_edge(4, 41, weight=0.75)
#G.add_edge(4, 42, weight=0.4)
G.add_edge(4, 42, weight=0.4)
#G.add_edge(4, 44, weight=0.4)
#G.add_edge(4, 45, weight=0.4)
#G.add_edge(6, 8, weight=0.4)
G.add_edge(5, 6, weight=1.5)
G.add_edge(5, 50, weight=0.4)
G.add_edge(5, 51, weight=0.8)
G.add_edge(5, 52, weight=1)
#G.add_edge(5, 53, weight=0.4)
G.add_edge(6, 52, weight=0.4)
G.add_edge(6, 60, weight=1.2)
G.add_edge(6, 61, weight=0.5)
G.add_edge(7, 70, weight=2)
G.add_edge(7, 71, weight=0.8)
G.add_edge(7, 72, weight=0.4)
#G.add_edge(8, 80, weight=0.4)

# Calculate node importance with normalization
normalized_importance_scores = calculate_node_importance_and_normalize(G)

# Print normalized importance scores
for node, score in sorted(normalized_importance_scores.items(), key=lambda item: item[1], reverse=True):
    print(f"Node {node} normalized importance score: {score:.2f}")
