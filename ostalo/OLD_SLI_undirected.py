#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 21:23:03 2024

@author: sanda
"""

import networkx as nx
import itertools

def calculate_node_importance_and_normalize(G):
    # Find all cycles in the graph, treating it as undirected for cycle finding
    cycle_basis = nx.cycle_basis(G)

    # Initialize a counter for each edge found in cycles
    edge_cycle_count = {}
    for cycle in cycle_basis:
        for i in range(len(cycle)):
            next_node = cycle[(i + 1) % len(cycle)]
            edge = frozenset({cycle[i], next_node})  # Use frozenset for undirected edge representation
            edge_cycle_count[edge] = edge_cycle_count.get(edge, 0) + 1

    # Calculate node strengths
    node_strength = {}
    for node in G.nodes():
        node_strength[node] = sum(data['weight'] for _, _, data in G.edges(node, data=True))

    # Calculate importance scores incorporating cycle participation and structural properties
    importance_scores = {}
    for node in G.nodes():
        importance_score = 0
        for neighbor in G.neighbors(node):
            edge = frozenset({node, neighbor})
            lambda_value = edge_cycle_count.get(edge, 0) + 1  # Number of cycles the edge participates in + 1
            u = node_strength[node] + node_strength[neighbor] - 2 * G[node][neighbor]['weight']
            z = node_strength[node] / (node_strength[node] + node_strength[neighbor])
            importance_score += lambda_value * u * z * G[node][neighbor]['weight']
            print((node, neighbor), lambda_value)
        
        importance_scores[node] = importance_score + node_strength[node]

    # Normalize the scores
    total_importance = sum(importance_scores.values())
    normalized_scores = {node: (score / total_importance * 100) for node, score in importance_scores.items()}

    return normalized_scores


#G = nx.Graph()

G = nx.Graph()
weighted_edges = [
    (1, 2, 3),  (1, 4, 3), (2, 1, 0.8), (4, 3, 1.4), (1, 5, 2), (1, 6, 2.5), (1, 7, 0.4),  (2, 20, 0.8), (2, 22, 0.7), (3, 1, 1.75), (3, 31, 1), (3, 40, 0.5),  (4, 41, 0.75), (4, 2, 0.4), (5, 52, 1), (5, 51, 0.8), (5, 50, 0.4), (6, 5, 1.5), (6, 20, 1.5),(6, 61, 0.5), (6, 60, 1.2), (7, 72, 0.4), (7, 70, 2), (21, 2, 0.8), (23, 2, 1), (32, 3, 0.4), (30, 3, 0.6), (40, 4, 1.5), (42, 4, 0.4), (52, 6, 0.4), (60, 2, 1.5), (71, 7, 0.8), (7, 73, 1), (73, 7, 1), (73, 732, 0.5), (731, 73, 0.5), (733, 73, 0.5)
]
G.add_weighted_edges_from(weighted_edges)


"""
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

"""

# Calculate node importance with normalization
normalized_importance_scores = calculate_node_importance_and_normalize(G)

# Print normalized importance scores
for node, score in sorted(normalized_importance_scores.items(), key=lambda item: item[1], reverse=True):
    print(f"Node {node} normalized importance score: {score:.2f}")
