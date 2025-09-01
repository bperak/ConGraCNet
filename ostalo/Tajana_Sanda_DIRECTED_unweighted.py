#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 09:49:37 2024

@author: sanda
"""

import networkx as nx
import matplotlib.pyplot as plt
#import scipy

#g = nx.DiGraph()


#G = nx.Graph()

G = nx.DiGraph()

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
G.add_edge(1, 2, weight=3)
G.add_edge(1, 3, weight=1.75)
G.add_edge(1, 4, weight=3)
#G.add_edge(4, 10, weight=0.4)
#G.add_edge(1, 5, weight=2)
#G.add_edge(1, 6, weight=2.5)
#G.add_edge(1, 7, weight=0.4)
#G.add_edge(2, 4, weight=0.4)
#G.add_edge(2, 20, weight=0.8)
#G.add_edge(2, 21, weight=2)
#G.add_edge(2, 22, weight=0.7)
#G.add_edge(2, 23, weight=1)
#G.add_edge(2, 60, weight=1.5)
#G.add_edge(3, 8, weight=0.4)
#G.add_edge(3, 4, weight=1.4)
#G.add_edge(3, 40, weight=0.5)
#G.add_edge(4, 2, weight=0.4)
#G.add_edge(3, 30, weight=0.6)
#G.add_edge(31, 3, weight=1)
#G.add_edge(3, 32, weight=0.4)
#G.add_edge(4, 40, weight=1.5)
#G.add_edge(4, 41, weight=0.75)
#G.add_edge(4, 42, weight=0.4)
#G.add_edge(42, 4, weight=0.4)
#G.add_edge(4, 44, weight=0.4)
#G.add_edge(4, 45, weight=0.4)
#G.add_edge(6, 8, weight=0.4)
#G.add_edge(5, 6, weight=1.5)
#G.add_edge(5, 50, weight=0.4)
#G.add_edge(5, 51, weight=0.8)
#G.add_edge(5, 52, weight=1)
#G.add_edge(5, 53, weight=0.4)
#G.add_edge(6, 52, weight=0.4)
#G.add_edge(6, 60, weight=1.2)
#G.add_edge(6, 61, weight=0.5)
#G.add_edge(7, 70, weight=2)
#G.add_edge(7, 71, weight=0.8)
#G.add_edge(7, 72, weight=0.4)
#G.add_edge(8, 80, weight=0.4)

#G.add_edge(131,673,weight=0.673)
#G.add_edge(131,201,weight=0.201)
#G.add_edge(131,303,weight=20)
#G.add_edge(673,96,weight=96)
#G.add_edge(673,205,weight=44)
#G.add_edge(673,110,weight=7)
#G.add_edge(201,96,weight=96)
#G.add_edge(201,232,weight=10)


plt.figure(figsize=(10,5))
pos = nx.planar_layout(G)
nx.draw(G, 
        pos=pos,
        node_size=800, 
        with_labels=True, 
        node_color='y')

#fig=plt.figure(figsize=(10,10))

#pos = nx.spring_layout(G, seed=6)#_layout(G)

weights = nx.get_edge_attributes(G,'weight').values()

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

G.to_directed(as_view=False)



#Plot full graph
#plt.subplot(211)
#plt.title('Full graph')
#labels_g = nx.get_edge_attributes(G,'weight')
#pos_g=nx.circular_layout(G)
#nx.draw_networkx_edge_labels(G,pos_g,edge_labels=labels_g)
#nx.draw(G,pos=pos_g,with_labels=True)

def check_neighbor_weights(g,nodes):
  subg=nx.Graph() #Create subgraph

  for n in nodes:
    subg.add_node(n)
    neighbors=g.neighbors(n) #Find all neighbors of node n
    for neighs in neighbors:
      if g[n][neighs]['weight']<50: #Check if the weigh t is below 50
        subg.add_edge(n,neighs,weight=g[n][neighs]['weight'])
  return subg

#subg=check_neighbor_weights(G,[131,201]) #Returns subgraph of interest

#plt.subplot(212)
#plt.title('subgraph')
#labels_subg = nx.get_edge_attributes(subg,'weight')
#pos_subg=nx.circular_layout(subg)
#nx.draw_networkx_edge_labels(subg,pos=pos_subg,edge_labels=labels_subg)
#nx.draw(subg,pos=pos_subg,with_labels=True)

plt.show()