#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 20:03:51 2024

@author: sanda
"""

#If you use the introduced Semi-Local Integration Measure, we kindly ask you to cite our article Semi-Local Integration Measure of Node Importance, 
#MDPI Mathematics Special Issue New Trends in Graph and Complexity Based Data Analysis and Processing.

import networkx as nx
from matplotlib import pyplot as plt

import itertools

import pandas as pd

# construct an example graph G
G = nx.Graph()

cycles = [[1, 2, 4], [1, 3, 4], [1, 5, 6], [3, 4, 40], [5, 6, 52]]
for cycle in cycles:
  nx.add_cycle(G, cycle)
cycles = [[1, 2, 4], [1, 3, 4], [1, 5, 6], [3, 4, 40], [5, 6, 52]]
for cycle in cycles:
  nx.add_cycle(G, cycle)

edges_tuple = [(1, 2, 3),(1, 3, 1.75),(1, 4, 3),(1, 5, 2),(1, 6, 2.5),(1, 7, 0.4),(2, 4, 0.4),(2, 20, 0.8),(2, 21, 2),(2, 22, 0.7),(2, 23, 1),(2, 60, 1.5),(3, 4, 1.4),(3, 40, 0.5),(3, 30, 0.6),(3, 31, 1),(3, 32, 0.4),(4, 40, 1.5),(4, 41, 0.75),(4, 42, 0.4),(5, 6, 1.5),(5, 50, 0.4),(5, 51, 0.8),(5, 52, 1),(6, 52, 0.4),(6, 60, 1.2),(6, 61, 0.5),(7, 70, 2),(7, 71, 0.8),(7, 72, 0.4)]
for edge in edges_tuple:
    G.add_edge(edge[0], edge[1], weight=edge[2])

# Set weighted_degree attribute on nodes
weights = nx.get_edge_attributes(G,'weight').values()
weighted_degree =  dict((x,y) for x,y in (G.degree(weight='weight')))
nx.set_node_attributes(G, weighted_degree, "weighted_degree")


def sli_importance(G, **kwargs):
    '''
    The algorithm takes graph G and returns the importance of the nodes as a list of floating values
    kwargs: 
        igraph (default False), if True transforms G from igraph to Networkx
        normalize (default True), if False returns non-normalized values from a list of SLI importance values
    '''
    # G introduced in igraph module is transformed into networkx module. If you want to introduce this feature you have to import igraph module
    if kwargs.get('igraph')==True:
        # import igraph
        g=G.to_networkx()
    else:
        g=G
    
    #Detection of basic cycles
    cycle = nx.cycle_basis(g.to_undirected()) #Will contain: node_value  -> sum_of_degrees
    degrees = {}    
    dict1 = {}
    for connect in (nx.edges(g)):
        degrees[connect[0]] = degrees.get(connect[0], 0) + g.degree(connect[1])
        degrees[connect[1]] = degrees.get(connect[1], 0) + g.degree(connect[0])

    #For each edge we get if a basic cycle contains it
        for cicl in cycle:
            in_path = lambda e, path: (e[0], e[1]) in path or (e[1], e[0]) in path
            cycle_to_path = lambda path: list(zip(path+path[:1], path[1:] + path[:1]))
            in_a_cycle = lambda e, cycle: in_path(e, cycle_to_path(cycle))
            in_any_cycle = lambda e, g: any(in_a_cycle(e, c) for c in nx.cycle_basis(g))

        counter_connect=0
        
    #For each edge the number of basic cycles that contain it is calculated
        if in_any_cycle(connect, g)==True:
            for cicl in (cycle):
                c=set(cicl)
                set1=set(connect)
                set2=set(c)
                is_subset = set1.issubset(set2)
                if is_subset==True:
                    counter_connect+=1
    
        dict1[connect] = counter_connect
        dict1[list(itertools.permutations(connect))[1]] = counter_connect
   
    
    SLI_importance = []
    for node in nx.nodes(g):
        sum = 0
        for neigh in g.neighbors(node):           
            edge_weight= g.get_edge_data(node,neigh)['weight']
            nodeWeight_node = g.nodes[node]["weighted_degree"]
            nodeWeight_neigh =  g.nodes[neigh]["weighted_degree"]
            p = dict1[(node, neigh)] 
            u = (nodeWeight_node+ nodeWeight_neigh - 2* edge_weight)
            lambd = p+1
            z = nodeWeight_node/ (nodeWeight_node+ nodeWeight_neigh)*edge_weight
            I = u*lambd*z
            sum = sum + I
        SLI_importance.append(sum + nodeWeight_node)
    SLI_importance= pd.Series(SLI_importance)
    
    
    # SLI values non-normalized
    if kwargs.get('normalize') == False:
        SLI_importance_result = SLI_importance
    # SLI values normalized as default
    else:
        SLI_importance_normalized = SLI_importance/SLI_importance.sum()*100
        SLI_importance_result = SLI_importance_normalized 
    return SLI_importance_result