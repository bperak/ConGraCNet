#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 02:00:24 2024

@author: sanda
"""

import networkx as nx

def find_cycles_for_all_edges(G):
    """
    For each edge in the graph, find and count the number of cycles it is part of.
    
    Parameters:
    - G: A directed graph (nx.DiGraph)
    
    Returns:
    - edge_cycle_counts: A dictionary where keys are edges (as tuples) and values are counts of cycles they are part of.
    - edge_cycles: A dictionary where keys are edges and values are the specific cycles they are part of.
    """
    cycles = list(nx.simple_cycles(G))
    edge_cycles = {}  # Stores specific cycles each edge is part of
    edge_cycle_counts = {}  # Stores counts of cycles for each edge

    for cycle in cycles:
        # Convert cycle to a list of directed edges
        cycle_edges = list(zip(cycle, cycle[1:] + cycle[:1]))
        for edge in cycle_edges:
            if edge in edge_cycles:
                edge_cycles[edge].append(cycle)
            else:
                edge_cycles[edge] = [cycle]

    # Count the number of cycles each edge is part of
    edge_cycle_counts = {edge: len(cycles) for edge, cycles in edge_cycles.items()}

    return edge_cycle_counts, edge_cycles

def main():
    # Create a directed graph object
    G = nx.DiGraph()
    
    # Define weighted edges to add to the graph
    weighted_edges = [
        (1, 2, 3), (2, 1, 0.8), (4, 3, 1.4), (1, 4, 3), (1, 5, 2), (1, 6, 2.5), (1, 7, 0.4),  (2, 20, 0.8), (2, 22, 0.7), (3, 1, 1.75), (3, 31, 1), (3, 40, 0.5),  (4, 41, 0.75), (4, 2, 0.4), (5, 52, 1), (5, 51, 0.8), (5, 50, 0.4), (6, 5, 1.5), (6, 20, 1.5),(6, 61, 0.5), (6, 60, 1.2), (7, 72, 0.4), (7, 70, 2), (21, 2, 0.8), (23, 2, 1), (32, 3, 0.4), (30, 3, 0.6), (40, 4, 1.5), (42, 4, 0.4), (52, 6, 0.4), (60, 2, 1.5), (71, 7, 0.8), (7, 73, 1), (73, 7, 1), (73, 732, 0.5), (731, 73, 0.5), (733, 73, 0.5)
    ]
    G.add_weighted_edges_from(weighted_edges)

    # Find cycles for all edges
    edge_cycle_counts, edge_cycles = find_cycles_for_all_edges(G)

    # Print results
    for edge, count in edge_cycle_counts.items():
        print(f"Edge {edge} is part of {count} cycle(s).")
        
    for edge, cycles in edge_cycles.items():
        print(f"Edge {edge} is part of the following cycle(s):")
        for cycle in cycles:
            print(cycle)
    
    for edge, count in edge_cycle_counts.items():
        print(f"Edge {edge}: {count} cycle(s)")

# Execute the main function
main()
