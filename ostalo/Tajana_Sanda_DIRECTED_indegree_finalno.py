#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 21:06:37 2024

@author: sanda
"""


import copy
import networkx as nx
#import grinpy as gp
#from grinpy.invariants.clique import *  # noqa
import numpy
import itertools
from matplotlib import pyplot as plt
from scipy import *


G = nx.DiGraph()

#lista_cvorova = [(1, 2), (1, 40), (2, 3), (3, 4), (4, 1), (1, 10), (10, 20), (20, 30), (30, 10), (20, 1)]

#nodes = G.add_weighted_edges_from([(1, 2, 0.5), (1, 40, 0.75), (2, 3, 1), (3, 4, 0.25), (4, 1, 1.25), (1, 10, 1), (10, 20, 1), (20, 30, 0.5), (30, 10, 0.75), (20, 1, 0.6)])


nodes = G.add_weighted_edges_from([(1, 2, 3), (1, 4, 3), (1, 5, 2), (1, 6, 2.5), (1, 7, 0.4), (2, 1, 0.8), (2, 20, 0.8), (2, 22, 0.7), (3, 1, 1.75), (3, 31, 1), (3, 40, 0.5), (4, 3, 1.4), (4, 41, 0.75), (4, 2, 0.4), (5, 52, 1), (5, 51, 0.8), (5, 50, 0.4), (6, 5, 1.5), (6, 20, 1.5),(6, 61, 0.5), (6, 60, 1.2), (7, 72, 0.4), (7, 70, 2), (21, 2, 0.8), (23, 2, 1), (32, 3, 0.4), (30, 3, 0.6), (40, 4, 1.5), (42, 4, 0.4), (52, 6, 0.4), (60, 2, 1.5), (71, 7, 0.8), (7, 73, 1), (73, 7, 1), (73, 732, 0.5), (731, 73, 0.5), (733, 73, 0.5)]) 

b=nx.betweenness_centrality(G)
print("*betweenness*", b)   

p=nx.pagerank(G, alpha=0.9)
print("*pagerank*", p)  

#%%
ciklusi = list(nx.simple_cycles(G))

print("Simple cycles", ciklusi)
print("Broj jednostavnih ciklusa", len(ciklusi))

print("ciklusici 0", ciklusi[0])
print("ciklusici 1", ciklusi[1])
print("ciklusici 2", ciklusi[2])

print("Find cycles", nx.find_cycle(G))
#print(nx.find_cycle(G, 10))


print("Lista bridova",  nx.find_cycle(G, G.nodes))

for i in range(0, len(ciklusi)):
    print("Find cycles", nx.find_cycle(G))
    
print(G.nodes)

def edge_in_cycle(edge, graph):
    u, v = edge
    basis = nx.cycle_basis(graph)
    edges = [zip(nodes,(nodes[1:]+nodes[:1])) for nodes in basis]
    found = False
    for cycle in edges:
        if (u, v) in cycle:
            found = True            
    return found

for i in range(0, len(ciklusi)):
    print(ciklusi[i])

print("G.edges", G.edges)

dict = {}

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

print("intersection", intersection((1, 2), ciklusi[0]), len(intersection((1, 2), ciklusi[0])))
#%%

for edge in G.edges:
    counter_veze = 0
    #for ciklus in ciklusi:
     #   print("test ciklusa", edge, ciklus)
    for i in range(0, len(ciklusi)):
            #print(ciklusi[i])
        lista_bridova = nx.find_cycle(G, ciklusi[i])
        print("Sto gledamo?", lista_bridova, i)
        #for j in range(0, len(lista_bridova)):
            #R = nx.intersection(edge, lista_bridova[j])
        #print([i for i in ciklusi[i] if i in edge])
        set_2 = frozenset(edge)
        intersection = [x for x in ciklusi[i] if x in set_2]
        if(len(intersection)==2):
        #if(len(intersection(edge, ciklusi[i]))==2):
            #if (any('geeksforgeeks' in i for i in test_tuple)):
            counter_veze += 1
            print("Brid", edge, "jest u ciklusu", ciklusi[i], "counter_veze", counter_veze)             
            dict.update({edge : counter_veze})
            #.update({"three":3})
            #dict = copy.deepcopy(dict)
        else:
            print("Brid", edge, "nije u ciklusu", ciklusi[i], "counter_veze", counter_veze)
            dict.update({edge : counter_veze})
            #dict = copy.deepcopy(dict) 
    print(dict)
    
#%%

print(dict)

dict.update({(1, 2) : 1})

print(dict)

#%%
def common_out_neighbors(g, i, j):
    return set(g.successors(i)).intersection(g.successors(j))

def common_in_neighbors(g, i, j):
    return set(g.predecessors(i)).intersection(g.predecessors(j))
    
for node in G.nodes:
    print("OUT_Susjedi od", node, list(G.successors(node)))
    print("IN_Susjedi od", node, list(G.predecessors(node)))

for node in G.nodes:
    print("broj OUT_Susjeda od", node, "je", len(list(G.successors(node))))
    print("broj IN_Susjeda od", node, "je", len(list(G.predecessors(node))))

#%%
#SLI- DIRECTED#

nasa_directed_vaznost = []
 
dict1 = {}

dict2 = {}

dict4 = {}

#print("*****", G[1][2]["weight"])

edge_labels = nx.get_edge_attributes(G, 'weight')
print("*edge labels*", edge_labels)
print("*edge labels values*", edge_labels[(1, 2)])

nasa_directed_vaznost = []

for node in G.nodes:
    #print(node, list(G.predecessors(node)), "broj INsusjeda cvora je", len(list(G.predecessors(node))) )
    #print(node, list(G.successors(node)), "broj OUTsusjeda cvora je", len(list(G.successors(node))) )
    suma = 0
    strength = 0
    for susjed in G.predecessors(node):
        #print(node, susjed, edge_labels[(node, susjed)])
        #print()
        edge = (susjed, node)
        print("susjed: ", susjed, "node", node,  "odgovarajuci weight", G.get_edge_data(susjed, node)['weight'])#, edge_labels[(1, 40)])
        strength += G.get_edge_data(susjed, node)['weight']
        print("strength od", node, "je", strength)
        dict2.update({node : strength})
    print("node", node, "strength", strength)



dict2[21] = 0
dict2[30] = 0
dict2[32] = 0
dict2[23] = 0
dict2[71] = 0
dict2[42] = 0
dict2[731] = 0
dict2[733] = 0

print("Dict 2 INDEGREE", dict2)

print(len(dict2))

#%%

for node in G.nodes:
    #print(node, list(G.predecessors(node)), "broj INsusjeda cvora je", len(list(G.predecessors(node))) )
    #print(node, list(G.successors(node)), "broj OUTsusjeda cvora je", len(list(G.successors(node))) )
    suma = 0
    strength = 0
    for susjed in G.successors(node):
        #print(node, susjed, edge_labels[(node, susjed)])
        #print()
        edge = (node, susjed)
        print("node: ", node, "susjed", susjed, "odgovarajuci weight", G.get_edge_data(node, susjed)['weight'])#, edge_labels[(1, 40)])
        strength += G.get_edge_data(node, susjed)['weight']
        print("strength od", node, "je", strength)
        dict4.update({node : strength})
    
print(dict4)

print(len(dict4))
    
#%%
dict4.update({20 : 0})
dict4.update({22 : 0})
#dict4.update({23 : 0})
#dict4.update({30 : 0})
dict4.update({31 : 0})
#dict4.update({32 : 0})
dict4.update({41 : 0})
#dict4.update({42 : 0})
dict4.update({50 : 0})
dict4.update({52 : 0})
dict4.update({51 : 0})
dict4.update({61 : 0})
dict4.update({70 : 0})
#dict4.update({71 : 0})
dict4.update({72 : 0})
#dict4.update({731 : 0})
dict4.update({732 : 0})
#dict4.update({733 : 0})


print("Dict4 OUT DEGREE NOVI!", dict4)

print(len(dict4))

#%%

def get_number_of_elements(list):
    count = 0
    for element in list:
        count += 1
    return count

#%%

def zagradica(node):
    rezultat = dict2[node]
    return rezultat

for node in G.nodes:
    print("Node", node, "zagradica", zagradica(node))
    
#%%

print("Pocinje petlja koja nam treba za -")

#dict5 ={}

for node in G.nodes:
    print("******", node, list(G.predecessors(node)))
    #if len(G.predecessors(node)) != None and len(G.successors(node))!= None:
    #    zagradica = len(G.predecessors(node))*dict4[node] + len(G.successors(node))*dict2[node]
    #elif len(G.predecessors(node)) == 0:
     #   zagradica =  len(G.successors(node))*dict2[node]
    #else:
     #   zagradica = len(G.predecessors(node))*dict4[node]        
    ##zagradica = get_number_of_elements(G.predecessors(node))*dict4[node] + get_number_of_elements(G.successors(node))*dict2[node]
    ##print(get_number_of_elements(G.predecessors(node)), dict4[node], get_number_of_elements(G.successors(node)), dict2[node])
    #print("zagradica", zagradica)
    umnozak = 0
    """
    for x in list(G.successors(node)):   
        print("Vrh, susjed: ", node, ",",  x)
        #print("dict counter veze", dict[(node, x)])
        #dict2['key'] if 'key' in dict2 else 0
        zagrada = zagradica(node) + zagradica(x) - G.get_edge_data(node, x)['weight']#dict2[node]+dict2[x]-()*G.get_edge_data(node, x)['weight']
        print("zagrada", zagrada)
        print("counter veze", dict[(node, x)])
        print("counter veze uvecan za jedan", dict[(node, x)]+1)
        w = G.get_edge_data(node, x)['weight']
        print("weight", w)
        print("zagradica(node)", zagradica(node))
        omjer = zagradica(node)/(zagradica(node)+zagradica(x))
        print("omjer", omjer)
        print("Umnozak iz koraka", (dict[(node, x)]+1)*zagrada * w * omjer)
        umnozak += (dict[(node, x)]+1)*zagrada*w*omjer
        print("umnozak za", node, umnozak)
        #umnozak1 = umnozak + zagradica(node)
        #print("umnozak za", node, umnozak1)
        #print("deg od nodea", zagradica(node))
        dict3.update({node : umnozak})
        print(" ")
        #dict3.update({node : umnozak})
        #print(" ")
        #print(nx.nodes(G))
        #print('suma liste', sum(dict3))
        #normirana_lista = []
        suma = sum(dict3.values())
        norm = [i/suma*100 for i in dict3.values()]
        #*******************************************
    """
    for x in list(G.predecessors(node)):   
        print("Susjed, vrh: ", x, ",",  node)
            #print("dict counter veze", dict[(node, x)])
            #dict2['key'] if 'key' in dict2 else 0
        zagrada = zagradica(node) + zagradica(x) - 2*G.get_edge_data(x, node)['weight']
        print("zagrada", zagrada)
        print("counter veze", dict[(x, node)])
        print("counter veze uvecan za jedan", dict[(x, node)]+1)
        w = G.get_edge_data(x, node)['weight']
        print("weight", w)
        print("zagradica(node)", zagradica(node))
        omjer = zagradica(node)/(zagradica(node)+zagradica(x))
        print("omjer", omjer)
        print("Umnozak iz koraka", (dict[(x, node)]+1)*zagrada * w * omjer)
        umnozak += (dict[(x, node)]+1)*zagrada*w*omjer
        print("umnozak za", node, umnozak)
        #print("deg od nodea", zagradica(node))
        #umnozak1 = umnozak + zagradica(node)
        #print("umnozak za", node, umnozak1)
        dict3.update({node : umnozak})
        print(" ")
        #print(dict3.update({node : umnozak + zagradica(node)}))
            #print(nx.nodes(G))
            #print('suma liste', sum(dict3))
            #normirana_lista = []
        suma = sum(dict3.values())
        norm = [i/suma*100 for i in dict3.values()]
        #print(norm, len(norm))
        #print(G.nodes, len(G.nodes))
        #dict5 = dict(zip(G.nodes(), norm))
        #print(dict5)
        #umnozak = zagrada*w*omjer
    #umnozak = umnozak + umnozak
    #print("FINALE", umnozak)
    #novi_umnozak = 0
    #novi_umnozak += umnozak
    #print("Novi umnozak", novi_umnozak)

#for susjed in G.successors(node):
#    print(susjed, dict2[susjed])
        #if G.has_edge(node, susjed):
            #print("Dobili smo: ", (node, susjed))
        #else:
            #print("nee", (node, susjed))
    #for susjed in list(G.predecessors(node)):
      #  print(dict2(node))
        #strength = G.in_degree(susjed)
        #print("**strength**", G.predecessors.size(weight="weight"))
        #u = (G.in_degree(node)+G.in_degree(susjed)-2*G.get_edge_data(susjed, node)['weight'])*G.get_edge_data(susjed, node)['weight']
        #print("u = ", u)
        #w = G.get_edge_data(susjed, node)['weight']
        #print("w = ", w)
        #z = (G.in_degree(node))/(G.in_degree(node)+G.in_degree(susjed))
        #print("z = ", z)
        #rezultat0 = u*w
        #rezultat = u*w*z
        #print('u, w, z, rezultat0, rezultat:', u, w, z, rezultat0, rezultat)
        #suma = suma + rezultat
        #print("suma", suma)
        #suma = suma + rezultat
    #print(suma)
    #nasa_directed_vaznost.append(suma)
    #print("suma", suma)
    #nasa_directed_vaznost.append(suma + G.in_degree(node))
    #print(nx.nodes(G))
    #print('suma liste', sum(nasa_directed_vaznost))
    #normirana_lista = []
    #norm = [i/sum(nasa_directed_vaznost)*100 for i in nasa_directed_vaznost]

#print(norm)
        #print()
        #print(G[node][susjed])
        #weights = [G[node][susjed]['weight'] for node, susjed in G.edges()]
        #print("Node, susjed", node, susjed)
        
#for u, v in G.edges:
#    print(u, v, G[u][v]['weight'])

#%%



#%%

for node in G.nodes:
    #print(node, list(G.predecessors(node)), "broj INsusjeda cvora je", len(list(G.predecessors(node))) )
    #print(node, list(G.successors(node)), "broj OUTsusjeda cvora je", len(list(G.successors(node))) )
    suma = 0
    strength = 0
    for susjed in G.successors(node):
        #print(node, susjed, edge_labels[(node, susjed)])
        #print()
        edge = (node, susjed)
        print("node: ", node, "susjed", susjed, "odgovarajuci weight", G.get_edge_data(node, susjed)['weight'])#, edge_labels[(1, 40)])
        strength += G.get_edge_data(node, susjed)['weight']
        print("strength od", node, "je", strength)
        dict4.update({node : strength})
    #print("node", node, "strength", strength)

#print("Dict4 IN_DEGREE", dict4)

#dict4.update({23 : 0})
#dict4.update({21 : 0})
#dict4.update({30 : 0})
#dict4.update({32 : 0})
#dict4.update({42 : 0})
#dict4.update({71 : 0})
#dict4.update({731 : 0})
#dict4.update({733 : 0})

#dict4.update({20 : 0})
#dict4.update({21 : 0.8})
##dict4.update({22 : 0.7})
#dict4.update({23 : 0})
#dict4.update({30 : 0})
#dict4.update({31 : 1})
#dict4.update({32 : 0})
#dict4.update({41 : 0.75})
#dict4.update({42 : 0})
#dict4.update({50 : 0})
#dict4.update({51 : 0})
#dict4.update({61 : 0.5})
#dict4.update({70 : 0})
#dict4.update({71 : 0})
#dict4.update({72 : 0})
##dict4.update({731 : 0})
#dict4.update({732 : 0})
#dict4.update({733 : 0})


print("Dict4 OUT DEGREE NOVI!", dict4)

#%%
#list(G.predecessors(732))
#%%
#%%

#dict2[20] = 2.3
#dict2[22] = 0.7
#dict2[21] = 0
#dict2[30] = 0
#dict2[32] = 0
#dict2[23] = 0
#dict2[41] = 0.75 *provjeri*
#dict2[51] = 0.8
#dict2[50] = 0.4 
#dict2[61] = 0.5 *provjeri*
#dict2[42] = 0
#dict2[72] = 0.4 
#dict2[70] = 2 
#dict2[71] = 0
#dict2[31] = 1 *provjeri*
#dict2[731] = 0
#dict2[732] = 0.5 *provjeri*
#dict2[733] = 0

#%%


print(dict2)
print(len(dict2))

print("dict2 outdegree", dict2)

print("dict4 indegree", dict4)

print("dict3", dict3, len(dict3))

print(get_number_of_elements(G.predecessors(1)))
print(dict4[731])
print(get_number_of_elements(G.successors(1)))
print(dict2[731])



#%%

dict3 = {}

#Dobra stvar!
"""
for node in G.nodes:
    print("******", node, list(G.successors(node)))
    umnozak = 0
    for x in list(G.successors(node)):   
        print(node, x)
        print("dict counter veze", dict[(node, x)])
        #dict2['key'] if 'key' in dict2 else 0
        zagrada = dict2[node]+dict2[x]-G.get_edge_data(node, x)['weight']
        print("zagrada", zagrada)
        w = G.get_edge_data(node, x)['weight']
        print("weight", w)
        omjer = dict2[node]/(dict2[node]+dict2[x])
        print("omjer", omjer)
        print("Umnozak iz koraka", zagrada * w * omjer)
        umnozak += zagrada*w*omjer
        print("umnozak za", node, umnozak)
        dict3.update({node : umnozak})
        #umnozak = zagrada*w*omjer
    #umnozak = umnozak + umnozak
    #print("FINALE", umnozak)
    #novi_umnozak = 0
    #novi_umnozak += umnozak
    #print("Novi umnozak", novi_umnozak)
"""

def get_number_of_elements(list):
    count = 0
    for element in list:
        count += 1
    return count

#%%

print("dict2 indegree", dict2)

print("dict4 outdegree", dict4)

print("dict3", dict3, len(dict3))

print(get_number_of_elements(G.predecessors(1)))
print(dict4[731])
print(get_number_of_elements(G.successors(1)))
print(dict2[731])

#def zagradica(node):
 #   rezultat = get_number_of_elements(G.predecessors(node))*dict4[node] + get_number_of_elements(G.successors(node))*dict2[node]
 #   return rezultat

def zagradica(node):
    rezultat = dict2[node]
    return rezultat

print(zagradica(1))
#%%

print("prec", list(G.predecessors(1)))
print("succ", list(G.successors(1)))

#%%

print(dict2)

#%% 
for node in G.nodes:
    print("Node", node, "zagradica", zagradica(node))
    
#%%
print("Pocinje petlja koja nam treba za -")

#dict5 ={}

for node in G.nodes:
    print("******", node, list(G.predecessors(node)))
    #if len(G.predecessors(node)) != None and len(G.successors(node))!= None:
    #    zagradica = len(G.predecessors(node))*dict4[node] + len(G.successors(node))*dict2[node]
    #elif len(G.predecessors(node)) == 0:
     #   zagradica =  len(G.successors(node))*dict2[node]
    #else:
     #   zagradica = len(G.predecessors(node))*dict4[node]        
    ##zagradica = get_number_of_elements(G.predecessors(node))*dict4[node] + get_number_of_elements(G.successors(node))*dict2[node]
    ##print(get_number_of_elements(G.predecessors(node)), dict4[node], get_number_of_elements(G.successors(node)), dict2[node])
    #print("zagradica", zagradica)
    umnozak = 0
    """
    for x in list(G.successors(node)):   
        print("Vrh, susjed: ", node, ",",  x)
        #print("dict counter veze", dict[(node, x)])
        #dict2['key'] if 'key' in dict2 else 0
        zagrada = zagradica(node) + zagradica(x) - G.get_edge_data(node, x)['weight']#dict2[node]+dict2[x]-()*G.get_edge_data(node, x)['weight']
        print("zagrada", zagrada)
        print("counter veze", dict[(node, x)])
        print("counter veze uvecan za jedan", dict[(node, x)]+1)
        w = G.get_edge_data(node, x)['weight']
        print("weight", w)
        print("zagradica(node)", zagradica(node))
        omjer = zagradica(node)/(zagradica(node)+zagradica(x))
        print("omjer", omjer)
        print("Umnozak iz koraka", (dict[(node, x)]+1)*zagrada * w * omjer)
        umnozak += (dict[(node, x)]+1)*zagrada*w*omjer
        print("umnozak za", node, umnozak)
        #umnozak1 = umnozak + zagradica(node)
        #print("umnozak za", node, umnozak1)
        #print("deg od nodea", zagradica(node))
        dict3.update({node : umnozak})
        print(" ")
        #dict3.update({node : umnozak})
        #print(" ")
        #print(nx.nodes(G))
        #print('suma liste', sum(dict3))
        #normirana_lista = []
        suma = sum(dict3.values())
        norm = [i/suma*100 for i in dict3.values()]
        #*******************************************
    """
    for x in list(G.predecessors(node)):   
        print("Susjed, vrh: ", x, ",",  node)
            #print("dict counter veze", dict[(node, x)])
            #dict2['key'] if 'key' in dict2 else 0
        zagrada = zagradica(node) + zagradica(x) - 2*G.get_edge_data(x, node)['weight']#dict2[node]+dict2[x]-()*G.get_edge_data(node, x)['weight']
        print("zagrada", zagrada)
        print("counter veze", dict[(x, node)])
        print("counter veze uvecan za jedan", dict[(x, node)]+1)
        w = G.get_edge_data(x, node)['weight']
        print("weight", w)
        print("zagradica(node)", zagradica(node))
        omjer = zagradica(node)/(zagradica(node)+zagradica(x))
        print("omjer", omjer)
        print("Umnozak iz koraka", (dict[(x, node)]+1)*zagrada * w * omjer)
        umnozak += (dict[(x, node)]+1)*zagrada*w*omjer
        print("umnozak za", node, umnozak)
        #print("deg od nodea", zagradica(node))
        #umnozak1 = umnozak + zagradica(node)
        #print("umnozak za", node, umnozak1)
        dict3.update({node : umnozak})
        print(" ")
        #print(dict3.update({node : umnozak + zagradica(node)}))
            #print(nx.nodes(G))
            #print('suma liste', sum(dict3))
            #normirana_lista = []
        suma = sum(dict3.values())
        norm = [i/suma*100 for i in dict3.values()]
        #print(norm, len(norm))
        #print(G.nodes, len(G.nodes))
        #dict5 = dict(zip(G.nodes(), norm))
        #print(dict5)
        #umnozak = zagrada*w*omjer
    #umnozak = umnozak + umnozak
    #print("FINALE", umnozak)
    #novi_umnozak = 0
    #novi_umnozak += umnozak
    #print("Novi umnozak", novi_umnozak)

#for susjed in G.successors(node):
#    print(susjed, dict2[susjed])
        #if G.has_edge(node, susjed):
            #print("Dobili smo: ", (node, susjed))
        #else:
            #print("nee", (node, susjed))
    #for susjed in list(G.predecessors(node)):
      #  print(dict2(node))
        #strength = G.in_degree(susjed)
        #print("**strength**", G.predecessors.size(weight="weight"))
        #u = (G.in_degree(node)+G.in_degree(susjed)-2*G.get_edge_data(susjed, node)['weight'])*G.get_edge_data(susjed, node)['weight']
        #print("u = ", u)
        #w = G.get_edge_data(susjed, node)['weight']
        #print("w = ", w)
        #z = (G.in_degree(node))/(G.in_degree(node)+G.in_degree(susjed))
        #print("z = ", z)
        #rezultat0 = u*w
        #rezultat = u*w*z
        #print('u, w, z, rezultat0, rezultat:', u, w, z, rezultat0, rezultat)
        #suma = suma + rezultat
        #print("suma", suma)
        #suma = suma + rezultat
    #print(suma)
    #nasa_directed_vaznost.append(suma)
    #print("suma", suma)
    #nasa_directed_vaznost.append(suma + G.in_degree(node))
    #print(nx.nodes(G))
    #print('suma liste', sum(nasa_directed_vaznost))
    #normirana_lista = []
    #norm = [i/sum(nasa_directed_vaznost)*100 for i in nasa_directed_vaznost]

#print(norm)
        #print()
        #print(G[node][susjed])
        #weights = [G[node][susjed]['weight'] for node, susjed in G.edges()]
        #print("Node, susjed", node, susjed)
        
#for u, v in G.edges:
#    print(u, v, G[u][v]['weight'])

print("dict3", dict3)
print("dict3", len(dict3))

dict3[21] = 0
dict3[23] = 0
dict3[32] = 0
dict3[30] = 0
dict3[42] = 0
dict3[71] = 0
dict3[731] = 0
dict3[733] = 0

print("dict3", dict3)
print(len(dict3))

#%%

print(dict2)
print(dict3)


for key in dict2:
    if key in dict3:
        my_dict[key] = dict2[key] + dict3[key]
    else:
        pass
         
print(my_dict)

suma = sum(my_dict.values())
norm = [i/suma*100 for i in my_dict.values()]

print(suma)
print(norm)

#%%

#lista1 = []

#for node in G.nodes:
#    list.append(lista1, zagradica(node))
    
#print("In degree stupnjevi", lista1)
#print(len(lista1))

#%%


#lista1.append()

#lista2 = []

#print("dict3", len(dict3))

#for values in dict3.values():
#   list.append(lista2, values)

#print("Ono sto je proslo kroz petlju, dict3", lista2)
#print("lista1", len(lista2))

#%%

#nova_lista = [x + y for x, y in zip(lista1, lista2)]

#print(nova_lista)

#print(list(zip(G.nodes, nova_lista)))

#my_dict = {}

#for k, v in zip(G.nodes, nova_lista):
#    my_dict[k] = v
#print(my_dict)

#suma = sum(my_dict.values())
#norm = [i/suma*100 for i in my_dict.values()]

#print(suma)
#print(norm)




#print(my_dict)

#%%

#print(dict3)
#print(dict2)

#for key in dict2:
#    if key in dict3:
#        my_dict[key] = dict2[key] + dict3[key]
#    else:
#        pass
         
#print(my_dict)


#%%

#dict33 = {}

#my_dict = {}
#for k, v in zip(G.nodes, nova_lista):
#    my_dict[k] = v
#print(my_dict)

#suma = sum(my_dict.values())
#norm = [i/suma*100 for i in my_dict.values()]

#print(suma)
#print(norm)

#for i in range(len(lista1)):
#    print(lista1[i]+lista2[i])
#

#%%
import pandas as pd

print("**Dict2 OUTDEGREE NOVI!**")

df2 = pd.DataFrame(list(dict3.items(), dict2.items(), my_dict.items()))
print(df2)


#%%

print(len(norm))

#%%
#len(dict3.values())
#len(norm)
#print(nx.nodes(G))
#print('suma liste', sum(dict3))
#normirana_lista = []
#norm = [i/sum(dict3.values())*100 for i in dict3.values()]
#print(norm)

#for node in G.nodes:
#    print(dict3[node]/sum(dict3.values())*100, "node value", dict3[node], sum(dict3.values()))

#print(sum(dict3.values()))
#%%
print("dict2", dict2)

print(dict2[1])

print("sta?", sum(dict3.values()))

print("dict3", dict3, len(dict3))

print(dict)

print("norm", norm)

#ciklusi = list(nx.simple_cycles(G))

#print("Simple cycles", ciklusi)
#%%


b=nx.betweenness_centrality(G)
print("*betweenness*", b)   

p=nx.pagerank(G, alpha=0.9)
print("*pagerank*", p)  

print("Find cycles", list(nx.simple_cycles(G)))

#G.get_edge_data(susjed, node)['weight']

#print(list(G.successors(1)))

'''
   
for node in nx.nodes(G):
    print(node, G.predecessors(node))
    # if node == veza[0]:
    suma = 0
    for susjed in G.predecessors(node):
            #edge_weight = G.get_edge_data(susjed, node)['weight']
            #print('susjed', G.get_edge_data(susjed, node)['weight'])
            #print('weight', edge_weight)
            #if node == veza[0]:
        print('ispis lala', node, len(G.predecessors(node)), susjed, len(G.predecessors(susjed)), dict1[(node, susjed)])
        print(nx.get_edge_attributes(G, 'weight'))
        edge_weight= G.get_edge_data(node, susjed)['weight']
        print('ew', edge_weight)
        u = (G.degree(node)+G.degree(susjed)-2*edge_weight)*edge_weight
        w = (dict1[(node, susjed)]+1)
        z = (G.degree(node))/(G.degree(node)+G.degree(susjed))
        rezultat0 = u*w
        rezultat = u*w*z
        print('u, w, z, rezultat0, rezultat:', u, w, z, rezultat0, rezultat)
        suma = suma + rezultat
        print(suma + len(G.predecessors(node)))
        nasa_directed_vaznost.append(suma + len(G.predecessors(node)))
    #print(nodes(G))
        print('suma liste', sum(nasa_directed_vaznost))
        #normirana_lista = []
        norm = [i/sum(nasa_directed_vaznost)*100 for i in nasa_directed_vaznost]
        print(norm)
        # suma_vaznosti = 0
        # for i in range(0, len(nasa_vaznost)+1):
        #     suma_vaznosti = suma_vaznosti + suma + G.degree(node)
    #return nasa_directed_vaznost

#print(sbb3_importance(G))
    
    
'''
    
