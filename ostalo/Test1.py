#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 20:23:17 2024

@author: sanda
"""

import copy
import itertools
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Create a directed graph object (NetworkX library function)
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

# Find all simple cycles in the graph
cycles = list(nx.simple_cycles(G))

print("Simple cycles:", cycles)
print("Number of simple cycles in the graph:", len(cycles))

# Function to check if an edge is part of a directed cycle
def edge_in_cycle(edge, graph):
    u, v = edge
    cycles = nx.simple_cycles(graph)
    return any((u, v) in cycle or (v, u) in cycle for cycle in cycles)

# Dictionary to store the count of cycles containing each edge
edge_cycle_count = defaultdict(int)

for edge in G.edges:
    for cycle in cycles:
        if edge_in_cycle(edge, G):
            edge_cycle_count[edge] += 1

print("Edge cycle count:", edge_cycle_count)

# Function to calculate the strength of a node based on its outgoing edges
def calculate_out_strength(node, graph):
    return sum(graph.get_edge_data(node, neighbor)['weight'] for neighbor in graph.successors(node))

# Calculate out-strength for each node
out_strength_dict = {node: calculate_out_strength(node, G) for node in G.nodes}

print("Out-strength dictionary:", out_strength_dict)

# Function to calculate the strength of a node based on its incoming edges
def calculate_in_strength(node, graph):
    return sum(graph.get_edge_data(neighbor, node)['weight'] for neighbor in graph.predecessors(node))

# Calculate in-strength for each node
in_strength_dict = {node: calculate_in_strength(node, G) for node in G.nodes}

print("In-strength dictionary:", in_strength_dict)

# Function to calculate the total strength of a node
def calculate_total_strength(node):
    return in_strength_dict[node] + out_strength_dict[node]

# Calculate total strength for each node
total_strength_dict = {node: calculate_total_strength(node) for node in G.nodes}

print("Total strength dictionary:", total_strength_dict)

# Function to calculate the SLI (Sum of Loop Intensity) for a node
def calculate_sli(node, graph):
    sli = 0
    for neighbor in graph.successors(node):
        factor1 = calculate_total_strength(node) + calculate_total_strength(neighbor) - 2 * graph.get_edge_data(node, neighbor)['weight']
        counter = edge_cycle_count[(node, neighbor)] + 1
        w = graph.get_edge_data(node, neighbor)['weight']
        ratio = calculate_total_strength(node) / (calculate_total_strength(node) + calculate_total_strength(neighbor))
        sli += counter * factor1 * w * ratio
    return sli

# Calculate SLI for each node
sli_dict = {node: calculate_sli(node, G) for node in G.nodes}

print("SLI dictionary:", sli_dict)

# Calculate the sum of SLI values for normalization
sli_sum = sum(sli_dict.values())

# Normalize SLI values
sli_normalized = {node: sli_value / sli_sum * 100 for node, sli_value in sli_dict.items()}

print("Normalized SLI dictionary:", sli_normalized)

# Calculate betweenness centrality and PageRank
betweenness_centrality = nx.betweenness_centrality(G)
print("Betweenness centrality:", betweenness_centrality)

pagerank = nx.pagerank(G, alpha=0.9)
print("PageRank:", pagerank)

#%%

def edge_in_cycle(graph, edge):
    """
    Check if the given directed edge is part of any cycle in the graph.

    Parameters:
        graph (networkx.DiGraph): The directed graph.
        edge (tuple): A tuple representing the directed edge (source, target).

    Returns:
        bool: True if the edge is part of any cycle, False otherwise.
    """
    u, v = edge
    cycles = nx.simple_cycles(graph)
    for cycle in cycles:
        if u in cycle and v in cycle:
            u_index = cycle.index(u)
            v_index = cycle.index(v)
            if (u_index < v_index and v_index - u_index == 1) or (v_index < u_index and u_index - v_index == 1):
                return True
    return False


print(edge_in_cycle(G, (2, 1)))

def count_cycles_with_edge(graph, edge):
    """
    Count how many cycles the given directed edge is a part of in the graph.

    Parameters:
        graph (networkx.DiGraph): The directed graph.
        edge (tuple): A tuple representing the directed edge (source, target).

    Returns:
        int: The number of cycles the edge is a part of.
    """
    u, v = edge
    count = 0
    for cycle in nx.simple_cycles(graph):
        if u in cycle and v in cycle:
            u_index = cycle.index(u)
            v_index = cycle.index(v)
            if (u_index < v_index and v_index - u_index == 1) or (v_index < u_index and u_index - v_index == 1):
                count += 1
    return count



print(count_cycles_with_edge(G, (1, 2)))
print(count_cycles_with_edge(G, (2, 1)))
