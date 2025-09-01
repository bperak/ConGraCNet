#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 19:36:32 2021

@author: sanda
"""

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
nx.add_cycle(G, [3, 4, 40])
#nx.add_cycle(G, [1, 2, 20, 6])
nx.add_cycle(G, [5, 6, 52])
#nx.add_cycle(G, [2, 60, 6, 20])
#nx.add_cycle(G, [1, 3, 8, 6])

#G.add_edge(1, 10, weight=0.4)
G.add_edge(1, 2, weight=1)
G.add_edge(1, 3, weight=1)
G.add_edge(1, 4, weight=1)
#G.add_edge(4, 10, weight=0.4)
G.add_edge(1, 5, weight=1)
G.add_edge(1, 6, weight=1)
G.add_edge(1, 7, weight=1)
G.add_edge(2, 4, weight=1)
G.add_edge(2, 20, weight=1)
G.add_edge(2, 21, weight=1)
G.add_edge(2, 22, weight=1)
G.add_edge(2, 23, weight=1)
G.add_edge(2, 60, weight=1)
#G.add_edge(3, 8, weight=0.4)
G.add_edge(3, 4, weight=1)
G.add_edge(3, 40, weight=1)
#G.add_edge(4, 2, weight=0.4)
G.add_edge(3, 30, weight=1)
G.add_edge(3, 31, weight=1)
G.add_edge(3, 32, weight=1)
G.add_edge(4, 40, weight=1)
G.add_edge(4, 41, weight=1)
#G.add_edge(4, 42, weight=0.4)
G.add_edge(4, 42, weight=1)
#G.add_edge(4, 44, weight=0.4)
#G.add_edge(4, 45, weight=0.4)
#G.add_edge(6, 8, weight=0.4)
G.add_edge(5, 6, weight=1)
G.add_edge(5, 50, weight=1)
G.add_edge(5, 51, weight=1)
G.add_edge(5, 52, weight=1)
#G.add_edge(5, 53, weight=0.4)
G.add_edge(6, 52, weight=1)
G.add_edge(6, 60, weight=1)
G.add_edge(6, 61, weight=1)
G.add_edge(7, 70, weight=1)
G.add_edge(7, 71, weight=1)
G.add_edge(7, 72, weight=1)
#G.add_edge(8, 80, weight=0.4)

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

def sbb3_importance(G, **kwargs):
    
    'G introduced in igraph module is transformed into networkx module'

    if kwargs.get('igraph')==True:
        import igraph
        g=G.to_networkx()
    else:
        g=G
    
    cycle = nx.cycle_basis(g.to_undirected()) # Will contain: { node_value => sum_of_degrees }
    degrees = {}    
    dict1 = {}
    for connect in (nx.edges(g)):
        degrees[connect[0]] = degrees.get(connect[0], 0) + g.degree(connect[1])
        degrees[connect[1]] = degrees.get(connect[1], 0) + g.degree(connect[0])

        for cicl in cycle:
            in_path = lambda e, path: (e[0], e[1]) in path or (e[1], e[0]) in path
            cycle_to_path = lambda path: list(zip(path+path[:1], path[1:] + path[:1]))
            in_a_cycle = lambda e, cycle: in_path(e, cycle_to_path(cycle))
            in_any_cycle = lambda e, g: any(in_a_cycle(e, c) for c in nx.cycle_basis(g))

        my_input = [] 
        counter_connect=0
        #veze = []
        #counters = []
        
        list2 = []
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
            nodeWeight_node = G.vs[node]["weighted_degree"]
            nodeWeight_neigh =  G.vs[neigh]["weighted_degree"]
            p = dict1[(node, neigh)] # broj ciklusa u kojem sudjeluje edge (node, susjed)
            u = (nodeWeight_node+ nodeWeight_neigh - 2* edge_weight)
            lambd = (dict1[(node, neigh)]+1)
            z = nodeWeight_node/ (nodeWeight_node+ nodeWeight_neigh)*edge_weight
            I = u*lambd*z
            sum = sum + I
        SLI_importance.append(sum + nodeWeight_node)
        
    if kwargs.get('normalize') == True:
        SLI_importance = normalize(SLI_importance)
    return SLI_importance


def normalize(list_values):
    #uzima Series i vraća series normaliziran
    list_values= pd.Series(list_values)
    sum= list_values.sum()
    normalized_list_values= list_values/sum*100
    return normalized_list_values


print(sbb3_importance(G))
#numpy.divide( sbb3_importance(G), sum(sbb3_importance(G)))
#print(sum(sbb3_importance(G)))
#print()

b=nx.betweenness_centrality(G)
print(b)

print(nx.pagerank(G, alpha=0.85, personalization=None, max_iter=100, tol=1e-06, nstart=None, dangling=None))
