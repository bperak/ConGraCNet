#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 12:11:28 2022

@author: sanda
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 11:23:40 2022

@author: sandažž

"""

import networkx as nx
import grinpy as grp
import numpy
import itertools
from matplotlib import pyplot as plt

G = nx.Graph()

#nx.add_cycle(G, [10,20,30])
nx.add_cycle(G, [3, 4, 4])
#nx.add_cycle(G, [1, 2, 2])
#nx.add_cycle(G, [1, 3, 4])
#nx.add_cycle(G, [1, 5, 6])
#nx.add_cycle(G, [3, 4, 40])
#nx.add_cycle(G, [1, 2, 20, 6])
#nx.add_cycle(G, [5, 6, 52])
#nx.add_cycle(G, [2, 60, 6, 20])
#nx.add_cycle(G, [1, 3, 8, 6])

#G.add_edge(1, 10, weight=0.4)
G.add_edge(1, 2, weight=1)
G.add_edge(1, 21, weight=1)
G.add_edge(1, 20, weight=1)
#G.add_edge(4, 10, weight=0.4)
G.add_edge(1, 19, weight=1)
G.add_edge(1, 18, weight=1)
G.add_edge(1, 17, weight=1)
G.add_edge(1, 22, weight=1)
G.add_edge(1, 23, weight=1)
G.add_edge(1, 3, weight=1)
G.add_edge(2, 27, weight=1)
G.add_edge(3, 24, weight=1)
G.add_edge(3, 26, weight=1)
#G.add_edge(3, 8, weight=0.4)
G.add_edge(3, 4, weight=1)
G.add_edge(4, 25, weight=1)
#G.add_edge(4, 2, weight=0.4)
G.add_edge(4, 5, weight=1)
G.add_edge(5, 6, weight=1)
G.add_edge(6, 7, weight=1)
G.add_edge(6, 8, weight=1)
G.add_edge(6, 9, weight=1)
#G.add_edge(4, 42, weight=0.4)
G.add_edge(6, 10, weight=1)
#G.add_edge(4, 44, weight=0.4)
#G.add_edge(4, 45, weight=0.4)
#G.add_edge(6, 8, weight=0.4)
G.add_edge(10, 16, weight=1)
G.add_edge(10, 15, weight=1)
G.add_edge(9, 14, weight=1)
G.add_edge(8, 13, weight=1)
#G.add_edge(5, 53, weight=0.4)
G.add_edge(8, 12, weight=1)
G.add_edge(8, 11, weight=1)
# G.add_edge(6, 61, weight=0.5)
# G.add_edge(7, 70, weight=2)
# G.add_edge(7, 71, weight=0.8)
# G.add_edge(7, 72, weight=0.4)
#G.add_edge(8, 80, weight=0.4)

pos = nx.spring_layout(G, seed=2)

#weights = nx.get_edge_attributes(G,'weight').values()

#nx.draw_networkx_edges(G,pos,alpha=0.5,edge_color='red')

#edge_labels=dict([((u,v,),d['weight'])

# for edge in G.edges(data='weight'):
#     nx.draw_networkx_edges(G, pos, width=list(weights))

#nx.draw_networkx_edge_labels(G,pos)

# edge_labels=dict([((u,v,),d['weight'])
# for u,v,d in G.edges(data=True)])
#nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='blue', bbox=None, ax=None, rotate=False)

nx.draw_networkx(G, pos)
plt.savefig("plot.png", dpi=1000)
plt.savefig("plot.pdf")

#nx.draw(G, pos, edges=edges, edge_color=colors, width=weights)


    
nx.draw_networkx_nodes(G, pos, node_size=75)

# weights = nx.get_edge_attributes(G,'weight').values()
# weighted_degree =  dict((x,y) for x,y in (G.degree(weight='weight')))
# print('weighted_degree\n', weighted_degree)
# nx.set_node_attributes(G, weighted_degree, "weighted_degree")

'''

def sbb3_importance(G):
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
            # print(node, G.degree(node), susjed, G.degree(susjed), dict1[(node, susjed)])
            # print(nx.get_edge_attributes(G, 'weight'))
            # edge_weight= G.get_edge_data(node,susjed)['weight']
            # print('ew', edge_weight)
            # u = (G.degree(node)+G.degree(susjed)-2*edge_weight)*edge_weight
            # w = (dict1[(node, susjed)]+1)
            # z = (G.degree(node))/(G.degree(node)+G.degree(susjed))
            # rezultat0 = u*w
            # rezultat = u*w*z
            # print('u, w, z, rezultat0, rezultat:', u, w, z, rezultat0, rezultat)
            # suma = suma + rezultat
            
            edge_weight= G.get_edge_data(node,susjed)['weight']
            nodeWeight_node = G.nodes[node]["weighted_degree"]
            print('nodeWeight_node', nodeWeight_node)
            nodeWeight_susjed =  G.nodes[susjed]["weighted_degree"]
            print('nodeWeight_susjed', nodeWeight_susjed)
            u = (nodeWeight_node+ nodeWeight_susjed - 2* edge_weight)
            print('u', u)
            lam = (dict1[(node, susjed)]+1)
            print('lam', lam)
            z = nodeWeight_node/ (nodeWeight_node+ nodeWeight_susjed)*edge_weight
            print('z', z)
            rezultat = u*lam*z
            print('rezultat', rezultat)
            suma = suma + rezultat
        print('suma', suma)
        nasa_vaznost.append(suma + nodeWeight_node)
        # print(nx.nodes(G))
        # print('suma liste', sum(nasa_vaznost))
        #normirana_lista = []
        norm = [i/sum(nasa_vaznost)*100 for i in nasa_vaznost]
        print(norm)
        # suma_vaznosti = 0
        # for i in range(0, len(nasa_vaznost)+1):
        #     suma_vaznosti = suma_vaznosti + suma + G.degree(node)
    return nasa_vaznost

print(sbb3_importance(G))
#numpy.divide( sbb3_importance(G), sum(sbb3_importance(G)))
#print(sum(sbb3_importance(G)))
#print()

'''

def sbb3_importance(G):
    'take graph and get the importance of the nodes as a list of float values'
    #g=G.to_networkx() 
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
        counter_veze=0
        veze = []
        counters = []
        
        list2 = []
        if in_any_cycle(veza, G)==True:
            for cicl in (ciklus):
                c=set(cicl)
                set1=set(veza)
                set2=set(c)
                is_subset = set1.issubset(set2)
                if is_subset==True:
                    counter_veze+=1
            
        dict1[veza] = counter_veze
        dict1[list(itertools.permutations(veza))[1]] = counter_veze  
    
    nasa_vaznost1 = []
    for node in nx.nodes(G):
        suma = 0
        for susjed in grp.neighborhood(G, node):   
            #print(node, G.degree(node), susjed, G.degree(susjed), dict1[(node, susjed)])
            #print(nx.get_edge_attributes(G, 'weight'))
            ##edge_weight= G.get_edge_data(node,susjed)['weight']
            #print('ew', edge_weight)
            u = (G.degree(node)+G.degree(susjed)-2)
            #w = (dict1[(node, susjed)]+1)
            #print(w)
            z = (G.degree(node))/(G.degree(node)+G.degree(susjed))
            #rezultat0 = u*w
            rezultat = u*z
            '''edge_weight= G.get_edge_data(node,susjed)['weight']
            #nodeWeight_node = G.vs[node]["weighted_degree"]
            #nodeWeight_susjed =  G.vs[susjed]["weighted_degree"]
            p = dict1[(node, susjed)] # broj ciklusa u kojem sudjeluje edge (node, susjed)
            u = (nodeWeight_node+ nodeWeight_susjed - 2* edge_weight)
            lam = (dict1[(node, susjed)]+1)
            z = nodeWeight_node/ (nodeWeight_node+ nodeWeight_susjed)*edge_weight
            rezultat = u*lam*z'''
            suma = suma + rezultat
        nasa_vaznost1.append(suma + G.degree(node))
    return nasa_vaznost1




# def normalize(list_values):
#     normalized_list_values = []
#     #uzima Series i vraća series normaliziran
#     list_values= [sbb3_importance(G)]
#     suma= sum(sbb3_importance(G))
#     normalized_list_values= normalized_list_values.append(sbb3_importance(G)/suma*100)
    
#     #[list_values]/suma*100
#     return normalized_list_values


print(sum(sbb3_importance(G)))

for i in range(0, len(sbb3_importance(G))):
    print(sbb3_importance(G)[i]/sum(sbb3_importance(G))*100)
#print(normalize(sbb3_importance(G)))

# #a=a.append(i)
# norm_lista = []

# for i in range(0, len(sbb3_importance(G))):
#     norm_lista = norm_lista.append(sbb3_importance[i]/sum(sbb3_importance(G))*100)

#print(norm_lista)
#print( for item in sbb3_importance(F)sum(sbb3_importance(G)))


def shishi_importance(G):
    'take graph and get the importance of the nodes as a list of float values'
    #g=G.to_networkx() 
    # st.write(g)
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
            
        #print('brid', veza, '--> broj ciklusa u kojima sudjeluje', counter_veze)
        dict1[veza] = counter_veze
        dict1[list(itertools.permutations(veza))[1]] = counter_veze
        #print(list(itertools.permutations(veza)))
        
    #print(dict1)   
    
    njihova_vaznost = []
    for node in nx.nodes(G):
        #print(node, grp.neighborhood(g, node))
        # if node == veza[0]:
        suma = 0
        for susjed in grp.neighborhood(G, node):
            #if node == veza[0]:
            #print(node, G.degree(node), susjed, G.degree(susjed), dict1[(node, susjed)])
            u = (G.degree(node)-dict1[(node, susjed)]-1)*(G.degree(susjed)-dict1[(node,susjed)]-1)
            #print(dict1[(node, susjed)])
            w = (dict1[(node, susjed)]/2+1)
            #print(w)
            z = (G.degree(node)-1)/(G.degree(node)+G.degree(susjed)-2)
            rezultat0 = u/w
            rezultat = u*z
            #print('u, w, z, rezultat0, rezultat:', u, w, z, rezultat0, rezultat)
            suma = suma + rezultat
        #print(suma + G.degree(node))
        njihova_vaznost.append(suma + G.degree(node))
    return njihova_vaznost

print(G.nodes())

print('Sbb', sbb3_importance(G))

print('Shishi', shishi_importance(G))


for i in range(0, len(sbb3_importance(G))):
    print(i, sbb3_importance(G)[i], shishi_importance(G)[i])