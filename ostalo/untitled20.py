#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 19:38:46 2024

@author: sanda
"""

import networkx as nx

def count_cycles_containing_edges(G):
    cycles = list(nx.simple_cycles(G))
    edge_cycle_count = {edge: 0 for edge in G.edges()}
    for edge in G.edges():
        for cycle in cycles:
            cycle_edges = list(zip(cycle, cycle[1:] + cycle[:1])) + list(zip(cycle[-1:] + cycle[:-1], cycle))
            if edge in cycle_edges:
                edge_cycle_count[edge] += 1
        edge_cycle_count[edge] += 1
    return edge_cycle_count

def calculate_node_strengths(G):
    out_strength = {}  # Correctly initialize out_strength
    for node in G.nodes:
        out_strength[node] = sum(G[node][succ]['weight'] for succ in G.successors(node))
    return out_strength  # Return out_strength

def bracket(node, out_strength):
    return out_strength[node]

def calculate_importance(G, edge_cycle_count, out_strength):
    importance_scores = {}
    for node in G.nodes():
        product = 0
        for x in list(G.successors(node)):  # Correctly iterating over successors for out_strength
            edge = (node, x)
            cycle_count = edge_cycle_count[edge] if edge in edge_cycle_count else 0
            factor = bracket(node, out_strength) + bracket(x, out_strength) - 2 * G.get_edge_data(*edge)['weight']
            w = G.get_edge_data(*edge)['weight']
            ratio = bracket(node, out_strength) / (bracket(node, out_strength) + bracket(x, out_strength))
            partial_product = (cycle_count + 1) * factor * w * ratio
            print("data", edge, cycle_count, out_strength[node], out_strength[x], w, ratio, partial_product)
            product += partial_product

        strength_sum = out_strength[node]
        adjusted_product = product + strength_sum

        importance_scores[node] = adjusted_product
        
        print(f"Total product for Node {node}: {product}")
        print(f"Total product for Node {node} + out-strength: {adjusted_product}\n")

    total = sum(importance_scores.values())
    normalized_scores = {node: score / total * 100 for node, score in importance_scores.items()}
    return normalized_scores

#brisati: (2, 1, 0.8), (4, 3, 1.4),
def main():
    G = nx.DiGraph()
    weighted_edges = [
        (1, 2, 3), (1, 4, 3), (1, 5, 2), (1, 6, 2.5), (1, 7, 0.4),  (2, 20, 0.8), (2, 22, 0.7), (3, 1, 1.75), (3, 31, 1), (3, 40, 0.5),  (4, 41, 0.75), (4, 2, 0.4), (5, 52, 1), (5, 51, 0.8), (5, 50, 0.4), (6, 5, 1.5), (6, 20, 1.5),(6, 61, 0.5), (6, 60, 1.2), (7, 72, 0.4), (7, 70, 2), (21, 2, 0.8), (23, 2, 1), (32, 3, 0.4), (30, 3, 0.6), (40, 4, 1.5), (42, 4, 0.4), (52, 6, 0.4), (60, 2, 1.5), (71, 7, 0.8), (7, 73, 1), (73, 7, 1), (73, 732, 0.5), (731, 73, 0.5), (733, 73, 0.5)
    ]
    G.add_weighted_edges_from(weighted_edges)

    edge_cycle_count = count_cycles_containing_edges(G)
    out_strength = calculate_node_strengths(G)  # Correctly capture out_strength
    importance_scores = calculate_importance(G, edge_cycle_count, out_strength)  # Correctly pass out_strength

    for node, score in importance_scores.items():
        print(f"Node {node} importance score: {score}%")

# Execute the main function
main()
