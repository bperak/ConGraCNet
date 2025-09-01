#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 23:28:04 2024

@author: sanda
"""

import networkx as nx
from itertools import combinations

def find_all_cycles(G, start_node):
    """Find all cycles in the undirected graph G starting at start_node."""
    visited = set()
    stack = [(start_node, [start_node])]
    while stack:
        current_node, path = stack.pop()
        visited.add(current_node)
        for neighbor in set(G.neighbors(current_node)) - set(path[-2:]):
            if neighbor == start_node:
                yield path
                continue
            if neighbor in visited:
                continue
            stack.append((neighbor, path + [neighbor]))

def all_cycles_in_graph(G):
    """Generate all cycles in an undirected graph G."""
    visited_nodes = set()
    for node in G.nodes():
        if node not in visited_nodes:
            for cycle in find_all_cycles(G, node):
                visited_nodes.update(cycle)
                yield cycle

def edge_cycle_counts(G):
    """Count how many cycles each edge in G is part of."""
    cycle_counts = {}
    for cycle in all_cycles_in_graph(G):
        # Get all edges in this cycle
        edges = set(frozenset([cycle[i], cycle[(i + 1) % len(cycle)]]) for i in range(len(cycle)))
        for edge in edges:
            cycle_counts[edge] = cycle_counts.get(edge, 0) + 1
    return cycle_counts

# Example usage
G = nx.Graph()
G.add_weighted_edges_from([
    (1, 2, {'weight': 3}), 
    (1, 4, {'weight': 3}), 
    (1, 5, {'weight': 2}), 
    (1, 6, {'weight': 2.5}), 
    (1, 7, {'weight': 0.4}), 
    (2, 20, {'weight': 0.8}),
    (2, 22, {'weight': 0.7}), 
    (2, 60, {'weight': 1.5}),
    (3, 1, {'weight': 1.75}), 
    (3, 31, {'weight': 1}), 
    (3, 40, {'weight': 0.5}),
    (4, 3, {'weight': 1.4}), 
    (4, 41, {'weight': 0.75}), 
    (4, 2, {'weight': 0.4}), 
    (5, 52, {'weight': 1}), 
    (5, 51, {'weight': 0.8}), 
    (5, 50, {'weight': 0.4}), 
    (6, 5, {'weight': 1.5}), 
    (6, 20, {'weight': 1.5}),
    (6, 61, {'weight': 0.5}), 
    (6, 60, {'weight': 1.2}), 
    (7, 72, {'weight': 0.4}), 
    (7, 70, {'weight': 2}), 
])

cycle_counts = edge_cycle_counts(G)
for edge, count in cycle_counts.items():
    print(f"Edge {set(edge)} is part of {count} cycle(s)")
    
cycles = [
    [1, 2, 20, 6], [1, 2, 20, 6, 5], [1, 2, 60, 6], [1, 2, 60, 6, 5], [1, 2, 4], [1, 2, 4, 3],
    [1, 4, 3], [1, 4, 2], [1, 4, 2, 20, 6], [1, 4, 2, 20, 6, 5], [1, 4, 2, 60, 6], [1, 4, 2, 60, 6, 5],
    [1, 5, 6], [1, 5, 6, 20, 2], [1, 5, 6, 20, 2, 4], [1, 5, 6, 20, 2, 4, 3], [1, 5, 6, 60, 2], [1, 5, 6, 60, 2, 4], [1, 5, 6, 60, 2, 4, 3],
    [1, 6, 5], [1, 6, 20, 2], [1, 6, 20, 2, 4], [1, 6, 20, 2, 4, 3], [1, 6, 60, 2], [1, 6, 60, 2, 4], [1, 6, 60, 2, 4, 3],
    [1, 3, 4], [1, 3, 4, 2], [1, 3, 4, 2, 20, 6], [1, 3, 4, 2, 20, 6, 5], [1, 3, 4, 2, 60, 6], [1, 3, 4, 2, 60, 6, 5]
]

def count_combined_edge_cycle_participation(cycles):
    edge_cycle_count = defaultdict(int)

    # First pass: count cycles for each edge direction independently
    for cycle in cycles:
        for i in range(len(cycle)):
            edge = (cycle[i], cycle[(i + 1) % len(cycle)])
            edge_cycle_count[edge] += 1

    # Second pass: combine counts for both directions of each edge
    combined_edge_cycle_count = defaultdict(int)
    for edge, count in edge_cycle_count.items():
        reverse_edge = (edge[1], edge[0])
        combined_count = edge_cycle_count[edge] + edge_cycle_count[reverse_edge]
        combined_edge_cycle_count[edge] = combined_count
        combined_edge_cycle_count[reverse_edge] = combined_count  # Ensure symmetry

    return combined_edge_cycle_count

# Count the combined cycles each edge (and its reverse) is a part of
combined_edge_cycle_count = count_combined_edge_cycle_participation(cycles)

# Print the combined count for each edge
for edge, combined_count in sorted(combined_edge_cycle_count.items()):
    print(f"Edge {edge} and its reverse are part of {combined_count} cycle(s) combined")