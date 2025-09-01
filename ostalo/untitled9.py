#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:47:17 2021

@author: sanda
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 18:08:37 2021

@author: sanda
"""

import networkx as nx
import grinpy as gp
import numpy
import itertools

G = nx.Graph()

nx.add_cycle(G, [1,2,3,4])
nx.add_cycle(G, [10,20,30])

e = (1, 10)

G.add_edge(1, 10)
G.add_edge(4, 10)
G.add_edge(1, 5)
G.add_edge(1, 6)
G.add_edge(2, 7)
G.add_edge(4, 8)
#G.add_edge(8, 30)

nx.draw(G, with_labels = True)

def sbb2_importance(G):
    'take graph and get the importance of the nodes as a list of float values'
    #g=G.to_networkx() 
    #st.write(G)
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
            #if node == veza[0]:
            print(node, G.degree(node), susjed, G.degree(susjed), dict1[(node, susjed)])
            u = (G.degree(node)+dict1[(node, susjed)])*(G.degree(susjed)+dict1[(node,susjed)])
            w = (dict1[(node, susjed)]/2+1)
            z = (G.degree(node))/(G.degree(node)+G.degree(susjed))
            rezultat0 = u*w
            rezultat = u*w*z
            print('u, w, z, rezultat0, rezultat:', u, w, z, rezultat0, rezultat)
            suma = suma + rezultat
        print(suma + G.degree(node))
        nasa_vaznost.append(suma + G.degree(node))
    return nasa_vaznost


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
                if len(cicl)==3:
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
    
    njihova_vaznost = []
    for node in nx.nodes(G):
        print(node, gp.neighborhood(G, node))
        # if node == veza[0]:
        suma = 0
        for susjed in gp.neighborhood(G, node):
            #if node == veza[0]:
            print(node, G.degree(node), susjed, G.degree(susjed), dict1[(node, susjed)])
            u = (G.degree(node)-dict1[(node, susjed)]-1)*(G.degree(susjed)-dict1[(node,susjed)]-1)
            w = (dict1[(node, susjed)]/2+1)
            z = (G.degree(node)-1)/(G.degree(node)+G.degree(susjed)-2)
            rezultat0 = u/w
            rezultat = u/w*z
            print('u, w, z, rezultat0, rezultat:', u, w, z, rezultat0, rezultat)
            suma = suma + rezultat
        print(suma + G.degree(node))
        njihova_vaznost.append(suma + G.degree(node))
    return njihova_vaznost

print(shishi_importance(G))









