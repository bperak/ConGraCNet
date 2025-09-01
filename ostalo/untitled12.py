#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 12:20:03 2024

@author: sanda
"""

import networkx as nx

def find_cycles_and_update_edge_counts(G):
    """
    For each edge in the graph, determine how many cycles it is part of and update a dictionary with this information.
    
    Parameters:
    - G: A directed graph (nx.DiGraph)
    
    Returns:
    - edge_cycle_counts: A dictionary mapping each edge to the number of cycles it is part of.
    """
    cycles = list(nx.simple_cycles(G))  # Find all simple cycles in the graph
    edge_cycle_counts = {}  # Initialize dictionary to store counts of cycles for each edge

    for edge in G.edges():
        counter_veze = 0  # Initialize cycle counter for the current edge
        for cycle in cycles:
            # Convert each cycle to a list of directed edges
            cycle_edges = list(zip(cycle, cycle[1:] + cycle[:1]))
            if edge in cycle_edges:  # Check if the current edge is part of this cycle
                counter_veze += 1
        
        edge_cycle_counts[edge] = counter_veze  # Update the dictionary with the count for the current edge

    return edge_cycle_counts

def main():
    # Create a directed graph object
    G = nx.Graph()
    
    # Define weighted edges to add to the graph (2, 1, 0.8), (4, 3, 1.4),
    weighted_edges = [
        (1, 2, 3), (1, 4, 3), (1, 5, 2), (1, 6, 2.5), (1, 7, 0.4), (2, 1, 0.8), (2, 20, 0.8),
        (2, 22, 0.7), (3, 1, 1.75), (3, 31, 1), (3, 40, 0.5), (4, 3, 1.4), (4, 41, 0.75),
        (4, 2, 0.4), (5, 52, 1), (5, 51, 0.8), (5, 50, 0.4), (6, 5, 1.5), (6, 20, 1.5),
        (6, 61, 0.5), (6, 60, 1.2), (7, 72, 0.4), (7, 70, 2), (21, 2, 0.8), (23, 2, 1),
        (32, 3, 0.4), (30, 3, 0.6), (40, 4, 1.5), (42, 4, 0.4), (52, 6, 0.4), (60, 2, 1.5),
        (71, 7, 0.8), (7, 73, 1), (73, 7, 1), (73, 732, 0.5), (731, 73, 0.5), (733, 73, 0.5)
    ]
    G.add_weighted_edges_from(weighted_edges)

    # Find cycles and update edge counts
    edge_cycle_counts = find_cycles_and_update_edge_counts(G)

    # Print the cycle counts for all edges
    for edge, count in edge_cycle_counts.items():
        print(f"Edge {edge}: {count} cycle(s)")

# Execute the main function
main()
