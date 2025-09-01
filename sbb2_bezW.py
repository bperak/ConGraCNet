#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 18:08:37 2021

@author: sanda
"""

import networkx as nx

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

nx.draw(G, with_labels = True)

def sbb2_importance(G):
    'take graph and get the importance of the nodes as a list of float values'
    #g=G.to_networkx() 
    #st.write(G)
    ciklus = nx.cycle_basis(G.to_undirected()) # Will contain: { node_value => sum_of_degrees }
    degrees = {}
    
    for veza in (nx.edges(G)):
        counter_veze=0
        degrees[veza[0]] = degrees.get(veza[0], 0) + G.degree(veza[1])
        degrees[veza[1]] = degrees.get(veza[1], 0) + G.degree(veza[0])
        print('brid:', veza)
        print('degree[veza[0]]', G.degree[veza[0]])
        print('degree[veza[1]]', G.degree[veza[1]])

        for cicl in ciklus:
            in_path = lambda e, path: (e[0], e[1]) in path or (e[1], e[0]) in path
            cycle_to_path = lambda path: list(zip(path+path[:1], path[1:] + path[:1]))
            in_a_cycle = lambda e, cycle: in_path(e, cycle_to_path(cycle))
            in_any_cycle = lambda e, g: any(in_a_cycle(e, c) for c in nx.cycle_basis(g))


    for veza in G.edges():
        
        if in_any_cycle(veza, G)==True:
            counter_veze=0
            for cicl in (ciklus):
                c=set(cicl)
                set1=set(veza)
                set2=set(c)
                is_subset = set1.issubset(set2)
                if is_subset==True:
                    counter_veze+=1
        else:
            counter_veze=0
        print(veza, counter_veze)
            
        u=(G.degree(veza[0])-counter_veze-1)*(G.degree(veza[1])-counter_veze-1)
        print('u=', u)
        lam = counter_veze/2+1
        print('lambda=', lam)
        I=u/lam
        print('I=', I)
        w=I*((G.degree(veza[0])-1)/(G.degree(veza[0])+G.degree(veza[1])-2))
        print('w=', w)

    zbroj_stupnjeva=0
    for i in nx.nodes(G):
        zbroj_stupnjeva+=G.degree(i)
    print('zbroj_stupnjeva', zbroj_stupnjeva)
        
    vaznost_sbb2_=[(round((G.degree(j)+degrees.get(j))/zbroj_stupnjeva, 5)) for j in nx.nodes(G)]
    # sbb_importance = []
    # for j in nx.nodes(g):
    #     sbb_importance.append((G.vs[j]["name"], round((g.degree(j)+degrees.get(j))/zbroj_stupnjeva, 5)))
    #     # st.write("Vaznost cvora {} je: ".format(G.vs[j]["name"]), round((g.degree(j)+degrees.get(j))/zbroj_stupnjeva, 5))
    # 'sbb_importance', sbb_importance
    return vaznost_sbb2_

print(sbb2_importance(G))