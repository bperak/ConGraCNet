#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 18:23:56 2021

@author: sanda
"""

import networkx as nx
import grinpy as gp
import numpy
import itertools
from matplotlib import pyplot as plt

G = nx.Graph()

#nx.add_cycle(G, [10,20,30])
#nx.add_cycle(G, [3, 4, 26])
nx.add_cycle(G, [1, 2, 4])
nx.add_cycle(G, [1, 3, 4])
nx.add_cycle(G, [1, 5, 6])
#nx.add_cycle(G, [1, 3, 8, 6])

#G.add_edge(1, 10, weight=0.4)
G.add_edge(1, 2, weight=0.4)
G.add_edge(1, 3, weight=0.4)
G.add_edge(1, 4, weight=4)
#G.add_edge(4, 10, weight=0.4)
G.add_edge(1, 5, weight=0.4)
G.add_edge(1, 6, weight=0.4)
G.add_edge(1, 7, weight=0.4)
G.add_edge(2, 20, weight=0.4)
G.add_edge(2, 21, weight=0.4)
G.add_edge(2, 22, weight=0.4)
#G.add_edge(3, 8, weight=0.4)
G.add_edge(3, 4, weight=0.4)
G.add_edge(4, 2, weight=0.4)
G.add_edge(3, 30, weight=0.4)
G.add_edge(3, 31, weight=0.4)
G.add_edge(4, 40, weight=0.4)
G.add_edge(4, 41, weight=0.4)
#G.add_edge(4, 42, weight=0.4)
G.add_edge(4, 43, weight=0.4)
G.add_edge(4, 44, weight=0.4)
G.add_edge(4, 45, weight=0.4)
#G.add_edge(6, 8, weight=0.4)
G.add_edge(5, 6, weight=0.4)
G.add_edge(5, 50, weight=0.4)
G.add_edge(5, 51, weight=0.4)
G.add_edge(5, 52, weight=0.4)
G.add_edge(5, 53, weight=0.4)
G.add_edge(6, 60, weight=0.4)
G.add_edge(7, 70, weight=0.4)
G.add_edge(7, 71, weight=0.4)
G.add_edge(7, 72, weight=0.4)
#G.add_edge(8, 80, weight=0.4)


pos = nx.spring_layout(G, seed=0)
nx.draw_networkx(G, pos)



for edge in G.edges(data='weight'):
    nx.draw_networkx_edges(G, pos, edgelist=[edge], width=edge[2])
    
nx.draw_networkx_nodes(G, pos, node_size=75)

def shishi_importance(G):
    'take graph and get the importance of the nodes as a list of float values'
    #g=G.to_networkx() 
    #st.write(g)
    ciklus = nx.cycle_basis(G.to_undirected()) # Will contain: { node_value => sum_of_degrees }
    degrees = {}
    
    dict1 = {}
    for veza in (nx.edges(G)):
        
        degrees[veza[0]] = degrees.get(veza[0], 0) + G.degree(veza[1])
        degrees[veza[1]] = degrees.get(veza[1], 0) + G.degree(veza[0])

        for cicl in ciklus:
            in_path = lambda e, path: (e[0], e[1]) in path or (e[1], e[0]) in path
            cycle_to_path = lambda path: list(zip(path+path[:1], path[1:] + path[:1]))
            in_a_cycle = lambda e, cycle: in_path(e, cycle_to_path(cycle))
            in_any_cycle = lambda e, g: any(in_a_cycle(e, c) for c in nx.cycle_basis(g))

        my_input = [] 
    #for veza in G.edges():
        counter_veze=0
        veze = []
        counters = []
        
        list2 = []
        if in_any_cycle(veza, G)==True:
            #counter_veze=0
            for cicl in (ciklus):
                #if len(cicl)==3:
                c=set(cicl)
                set1=set(veza)
                set2=set(c)
                is_subset = set1.issubset(set2)
                if is_subset==True:
                    counter_veze+=1
            
        print('brid', veza, '--> broj ciklusa u kojima sudjeluje', counter_veze)
        dict1[veza] = counter_veze
        dict1[list(itertools.permutations(veza))[1]] = counter_veze
        #print(list(itertools.permutations(veza)))
        
    #print(dict1)   
    
    nasa_vaznost = []
    for node in nx.nodes(G):
        print(node, gp.neighborhood(G, node))
        # if node == veza[0]:
        suma = 0
        for susjed in gp.neighborhood(G, node):
            #edge_weight = G.get_edge_data(susjed, node)['weight']
            #print('susjed', G.get_edge_data(susjed, node)['weight'])
            #print('weight', edge_weight)
            #if node == veza[0]:
            print(node, G.degree(node), susjed, G.degree(susjed), dict1[(node, susjed)])
            print(nx.get_edge_attributes(G, 'weight'))
            edge_weight= G.get_edge_data(node,susjed)['weight']
            print('ew', edge_weight)
                  #testaaa= node, susjed
                  #print((node, susjed))
                  #print(dict1[(node, susjed)])
                  #print(dict1)
                  #print(node, G.degree(node), susjed, G.degree(susjed), counter_veze)
            #u = (G.degree(node)+dict1[(node, susjed)])*(G.degree(susjed)+dict1[(node,susjed)])
            u = (G.degree(node)+dict1[(node, susjed)]+G.degree(susjed)-2*edge_weight)*edge_weight
            w = (dict1[(node, susjed)]/2+1)
            z = (G.degree(node))/(G.degree(node)+G.degree(susjed))
            rezultat0 = u*w
            rezultat = u*w*z
            print('u, w, z, rezultat0, rezultat:', u, w, z, rezultat0, rezultat)
            suma = suma + rezultat
        print(suma + G.degree(node))
        nasa_vaznost.append(suma + G.degree(node))
        print(nx.nodes(G))
    return nasa_vaznost

print(shishi_importance(G))