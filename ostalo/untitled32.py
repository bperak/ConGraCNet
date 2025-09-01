#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 23:19:26 2024

@author: sanda
"""

import networkx as nx
from collections import defaultdict

def find_all_cycles_and_edge_counts(G, start_node=None):
    if start_node is None:
        # Choose an arbitrary node from the graph
        start_node = list(G.nodes())[0]
    
    all_cycles = []
    edge_cycle_count = defaultdict(int)
    
    def dfs(current_path, visited):
        current_node = current_path[-1]
        for neighbor in G.neighbors(current_node):
            # Form an edge as a frozenset to handle undirected nature
            edge = frozenset([current_node, neighbor])
            if neighbor == start_node and len(current_path) > 2:
                all_cycles.append(current_path)
                for i in range(len(current_path)):
                    cycle_edge = frozenset([current_path[i], current_path[(i + 1) % len(current_path)]])
                    edge_cycle_count[cycle_edge] += 1
            elif neighbor not in visited:
                yield from dfs(current_path + [neighbor], visited | {neighbor})
    
    list(dfs([start_node], set([start_node])))
    return all_cycles, dict(edge_cycle_count)

def is_edge_in_cycles(edge, all_cycles):
    # Normalize edge representation
    edge_set = frozenset(edge)
    
    for cycle in all_cycles:
        # Create a set of edges for each cycle for easy comparison
        cycle_edges = [frozenset([cycle[i], cycle[(i + 1) % len(cycle)]]) for i in range(len(cycle))]
        
        if edge_set in cycle_edges:
            return True
    return False

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


# Correct usage of dict for printing or iterating
print("All cycles:")
for cycle in all_cycles:
    print(cycle)

print("\nEdge cycle counts:")
for edge, count in edge_cycle_count.items():  # Correct iteration over dict items
    print(f"Edge {set(edge)} is part of {count} cycle(s)")