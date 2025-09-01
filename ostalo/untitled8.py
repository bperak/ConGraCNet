#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 01:52:58 2024

@author: sanda
"""

import networkx as nx

def is_edge_in_cycle(G, edge, cycles):
    """
    Check if a given edge is part of any cycle in the graph.
    
    Parameters:
    - G: A directed graph (nx.DiGraph)
    - edge: A tuple representing an edge (source, target)
    - cycles: A list of cycles, where each cycle is represented as a list of nodes
    
    Returns:
    - is_in_cycle: Boolean indicating if the edge is part of any cycle
    """
    # Convert each cycle into a list of edges for easy comparison
    cycle_edges = [list(zip(cycle, cycle[1:] + cycle[:1])) for cycle in cycles]
    
    # Flatten the list of cycle edges
    flat_cycle_edges = [item for sublist in cycle_edges for item in sublist]
    
    # Check if the given edge is in the list of cycle edges
    is_in_cycle = edge in flat_cycle_edges
    
    return is_in_cycle

def main():
    # Create a directed graph object
    G = nx.DiGraph() 
    
    # Add weighted edges to the graph
    weighted_edges = [
        (1, 2, 3), (1, 4, 3), (1, 5, 2), (1, 6, 2.5), (1, 7, 0.4), (2, 1, 0.8), (2, 20, 0.8),
        (2, 22, 0.7), (3, 1, 1.75), (3, 31, 1), (3, 40, 0.5), (4, 3, 1.4), (4, 41, 0.75),
        (4, 2, 0.4), (5, 52, 1), (5, 51, 0.8), (5, 50, 0.4), (6, 5, 1.5), (6, 20, 1.5),
        (6, 61, 0.5), (6, 60, 1.2), (7, 72, 0.4), (7, 70, 2), (21, 2, 0.8), (23, 2, 1),
        (32, 3, 0.4), (30, 3, 0.6), (40, 4, 1.5), (42, 4, 0.4), (52, 6, 0.4), (60, 2, 1.5),
        (71, 7, 0.8), (7, 73, 1), (73, 7, 1), (73, 732, 0.5), (731, 73, 0.5), (733, 73, 0.5)
    ]
    G.add_weighted_edges_from(weighted_edges) 

    # Get simple cycles in the graph
    cycles = list(nx.simple_cycles(G))

    # Example usage of the is_edge_in_cycle function
    edge_to_check = (2, 1)  # Change this edge as needed
    edge_in_cycle = is_edge_in_cycle(G, edge_to_check, cycles)
    
    print(f"Edge {edge_to_check} is part of a cycle: {edge_in_cycle}")

# Execute the main function
main()
