#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 20:32:37 2024

@author: sanda
"""

import networkx as nx
#import grinpy as gp
import numpy
import itertools
from matplotlib import pyplot as plt

#G = nx.Graph()

G = nx.Graph()
weighted_edges = [
    (1, 2, 3), 
    (1, 4, 2.9), (4, 1, 0.1),
    (1, 5, 1.9), (5, 1, 0.1), 
    (1, 6, 2.4),  (6, 1, 0.1),
    (1, 7, 0.3), (7, 1, 0.1),
    (2, 1, 0.8),
    (2, 20, 0.7), (20, 2, 0.1), 
    (2, 22, 0.6), (22, 2, 0.1), 
    (3, 1, 1.65), (1, 3, 0.1),
    (3, 31, 0.9), (31, 1, 0.1),
    (3, 40, 0.4), (40, 3, 0.1),
    (4, 3, 1.3), (3, 4, 0.1), 
    (4, 41, 0.65), (41, 4, 0.1),
    (4, 2, 0.3), (2, 4, 0.1), 
    (5, 52, 0.9), (52, 5, 0.1), 
    (5, 51, 0.7), (51, 5, 0.1), 
    (5, 50, 0.3),  (50, 5, 0.1),
    (6, 5, 1.4), (5, 6, 0.1), 
    (6, 20, 1.4), (20, 6, 0.1), 
    (6, 61, 0.4), (61, 6, 0.1), 
    (6, 60, 1.1), (60, 6, 0.1),
    (7, 72, 0.3), (72, 7, 0.1), 
    (7, 70, 1.9), (70, 7, 0.1), 
    (21, 2, 0.7), (2, 21, 0.1),
    (23, 2, 0.9), (2, 23, 0.1),
    (32, 3, 0.3), (3, 32, 0.1), 
    (30, 3, 0.5), (3, 30, 0.1), 
    (40, 4, 1.4), (4, 40, 0.1), 
    (42, 4, 0.3), (4, 42, 0.1), 
    (52, 6, 0.4), (6, 52, 0.1), 
    (60, 2, 1.4), (2, 60, 0.1), 
    (71, 7, 0.7), (7, 71, 0.1),
    (7, 73, 1), 
    (73, 7, 1), 
    (73, 732, 0.4), (732, 73, 0.1), 
    (731, 73, 0.4), (73, 731, 0.1),
    (733, 73, 0.4), (73, 733, 0.1)
]
G.add_weighted_edges_from(weighted_edges)

pos = nx.spring_layout(G, seed=2)

weights = nx.get_edge_attributes(G,'weight').values()

#nx.draw_networkx_edges(G,pos,alpha=0.5,edge_color='red')

#edge_labels=dict([((u,v,),d['weight'])

for edge in G.edges(data='weight'):
    nx.draw_networkx_edges(G, pos, width=list(weights))

#nx.draw_networkx_edge_labels(G,pos)

# edge_labels=dict([((u,v,),d['weight'])
# for u,v,d in G.edges(data=True)])
#nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='blue', bbox=None, ax=None, rotate=False)

nx.draw_networkx(G, pos)
plt.savefig("plot.png", dpi=1000)
plt.savefig("plot.pdf")

#nx.draw(G, pos, edges=edges, edge_color=colors, width=weights)


    
nx.draw_networkx_nodes(G, pos, node_size=75)

weights = nx.get_edge_attributes(G,'weight').values()
#weighted_degree =  dict((x,y) for x,y in (G.degree(weight='weight')))
weighted_degree = {node: degree for node, degree in G.degree(weight='weight')}
print('weighted_degree\n', weighted_degree)
nx.set_node_attributes(G, weighted_degree, "weighted_degree")

import networkx as nx
import itertools

def sbb3_importance(G):
    # Calculate the cycle basis once
    cycle_basis = nx.cycle_basis(G.to_undirected())

    # Precompute edges in cycles
    edges_in_cycles = set()
    for cycle in cycle_basis:
        for i, node in enumerate(cycle):
            next_node = cycle[(i + 1) % len(cycle)]
            edges_in_cycles.add((node, next_node))
            edges_in_cycles.add((next_node, node))  # Add both directions for undirected graph

    # Calculate the number of cycles each edge is part of
    dict1 = {edge: 0 for edge in G.edges()}
    for edge in edges_in_cycles:
        dict1[edge] += 1  # Increment the count for edges part of a cycle

    importance_scores = []
    for node in G.nodes():
        sum_importance = 0
        weighted_degree_node = sum(G[node][nbr]['weight'] for nbr in G.neighbors(node))
        for neighbor in G.neighbors(node):
            edge_weight = G[node][neighbor]['weight']
            weighted_degree_neighbor = sum(G[neighbor][nbr]['weight'] for nbr in G.neighbors(neighbor))
            u = (weighted_degree_node + weighted_degree_neighbor - 2 * edge_weight)
            lam = dict1.get((node, neighbor), 0) + 1  # Use get to avoid KeyError, default to 0 if edge not found
            z = edge_weight * weighted_degree_node / (weighted_degree_node + weighted_degree_neighbor)
            sum_importance += u * lam * z
        total_importance = sum_importance + weighted_degree_node
        importance_scores.append(total_importance)

    # Normalize the importance scores
    total_sum = sum(importance_scores)
    #print("total suma", sum(importance_scores))
    normalized_scores = [score / total_sum * 100 for score in importance_scores]

    return normalized_scores

# Example usage
#G = nx.Graph()
# Add your edges and cycles as before

# Print the importance scores
importance_scores = sbb3_importance(G)
for i, score in enumerate(importance_scores, 1):
    print(f"Node {i} importance score: {score:.2f}")
    
    
