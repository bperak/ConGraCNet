#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 01:49:02 2024

@author: sanda
"""

import networkx as nx
import pandas as pd

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
    print("Simple cycles:", cycles)
    print("Number of simple cycles:", len(cycles))

    # Calculate strengths of nodes
    out_strengths = {node: sum(weight for _, _, weight in G.out_edges(node, data='weight')) for node in G.nodes}
    in_strengths = {node: sum(weight for _, _, weight in G.in_edges(node, data='weight')) for node in G.nodes}

    # Print out strengths
    print("Out-strengths:", out_strengths)
    print("In-strengths:", in_strengths)

    # Calculate degree for each node
    out_degrees = dict(G.out_degree())
    in_degrees = dict(G.in_degree())
    print("Out-degrees:", out_degrees)
    print("In-degrees:", in_degrees)

    # Compute betweenness centrality and pagerank
    betweenness_centrality = nx.betweenness_centrality(G)
    pagerank = nx.pagerank(G, alpha=0.9)

    print("Betweenness Centrality:", betweenness_centrality)
    print("PageRank:", pagerank)

if __name__ == "__main__":
    main()
