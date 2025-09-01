import copy
import networkx as nx
import numpy
import pandas as pd
import itertools
from matplotlib import pyplot as plt
from scipy import *

#Create a directed graph object (NetworkX library function)
G = nx.DiGraph() 

#We add weighted edges to the graph
#If we deal with the unweighted graph, then the third component in the triples is, for example 1, or it does not exist
nodes = G.add_weighted_edges_from([(1, 2, 3), (1, 4, 3), (1, 5, 2), (1, 6, 2.5), (1, 7, 0.4), (2, 1, 0.8), (2, 20, 0.8), (2, 22, 0.7), (3, 1, 1.75), (3, 31, 1), (3, 40, 0.5), (4, 3, 1.4), (4, 41, 0.75), (4, 2, 0.4), (5, 52, 1), (5, 51, 0.8), (5, 50, 0.4), (6, 5, 1.5), (6, 20, 1.5),(6, 61, 0.5), (6, 60, 1.2), (7, 72, 0.4), (7, 70, 2), (21, 2, 0.8), (23, 2, 1), (32, 3, 0.4), (30, 3, 0.6), (40, 4, 1.5), (42, 4, 0.4), (52, 6, 0.4), (60, 2, 1.5), (71, 7, 0.8), (7, 73, 1), (73, 7, 1), (73, 732, 0.5), (731, 73, 0.5), (733, 73, 0.5)]) 

#If we deal with the unweighted graph, then the third component in the triples is, for example 1, or it does not exist
#%%

#Lists of all cycles in the graph G; more information about the structure of the graph
cycls = list(nx.simple_cycles(G))

print("Simple cycles", cycls)
print("The number of simple cycles in the graph G", len(cycls))

#Function that investigates whether an edge is a part of the directed cycle or not

def edge_in_cycle(edge, graph):
    u, v = edge
    basis = nx.cycle_basis(graph)
    edges = [zip(nodes,(nodes[1:]+nodes[:1])) for nodes in basis]
    found = False
    for cycle in edges:
        if (u, v) in cycle:
            found = True            
    return found

print("G.edges", G.edges)

dict = {}

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


for edge in G.edges:
    counter = 0
    for i in range(0, len(cycls)):
        list_of_edges = nx.find_cycle(G, cycls[i])
        set_2 = frozenset(edge)
        intersection = [x for x in cycls[i] if x in set_2]
        if(len(intersection)==2):
            counter += 1
            print("The edge", edge, "is in the cycle", cycls[i], "counter", counter)             
            dict.update({edge : counter})
        else:
            print("The edge", edge, "is not in the cycle", cycls[i], "counter", counter)
            dict.update({edge : counter})
    #print(dict)
    
#%%

print(dict)

dict.update({(1, 2) : 1})
dict.update({(1, 6) : 1})

print(dict)

#%%
def common_out_neighbors(g, i, j):
    return set(g.successors(i)).intersection(g.successors(j))

def common_in_neighbors(g, i, j):
    return set(g.predecessors(i)).intersection(g.predecessors(j))
    
for node in G.nodes:
    print("OUT_neighbours of", node, list(G.successors(node)))
    print("IN_neighbours of", node, list(G.predecessors(node)))

for node in G.nodes:
    print("The number of OUT_neighbours of the node", node, "is", len(list(G.successors(node))))
    print("The number of IN_neighbours of the node", node, "is", len(list(G.predecessors(node))))

#%%
#SLI DIRECTED#

dsli_importance = []
 
dict1 = {}

dict2 = {}

dict4 = {}

edge_labels = nx.get_edge_attributes(G, 'weight')
print("*edge labels*", edge_labels)
print("*edge labels values*", edge_labels[(1, 2)])


for node in G.nodes:
    suma = 0
    strength = 0
    for neigh in G.successors(node):
        edge = (node, susjed)
        print(node, "The neighbour: ", neigh, "and corresponing weight", G.get_edge_data(node, neigh)['weight'])
        strength += G.get_edge_data(node, neigh)['weight']
        print("The strength of the node", node, "is", strength)
    dict2.update({node : strength})
    print("The node", node, "has strength", strength)

print("Dictionary of the outdegree centrality measure", dict2)

print(len(dict2))


#%%

for node in G.nodes:
    suma = 0
    strength = 0
    for neigh in G.predecessors(node):
        edge = (neigh, node)
        print("The neighbour: ", neigh, "node", node, "and corresponding weight", G.get_edge_data(neigh, node)['weight'])
        strength += G.get_edge_data(neigh, node)['weight']
        print("The strenght of the node", node, "is", strength)
    dict4.update({node : strength})
    print("node", node, "strength", strength)
    
print(dict4)

print(len(dict4))



#%%

print("INDEGREE: Dataframe")

df4 = pd.DataFrame(list(dict4.items()))
print(df4)


#%%

print("OUTDEGREE: Dataframe")


df2 = pd.DataFrame(list(dict2.items()))
print(df2)
#%%

def nodedeg(node):
    result = dict4[node] + dict2[node]
    return result

for node in G.nodes:
    print("node", node, nodedeg(node))


#%%
print("Pocinje petlja koja nam treba i za in i za out")

for node in G.nodes:
    print("******", node, list(G.successors(node)))
    pr = 0
    for x in list(G.successors(node)):   
        print("Node, neighbour: ", node, ",",  x)
        factor1 = nodedeg(node) + nodedeg(x) - 2*G.get_edge_data(node, x)['weight']
        print("Factor1", factor1)
        print("Counter", dict[(node, x)])
        print("Counter+1", dict[(node, x)]+1)
        w = G.get_edge_data(node, x)['weight']
        print("weight", w)
        print("zagradica(node)", nodedeg(node))
        ratio = nodedeg(node)/(nodedeg(node)+nodedeg(x))
        print("omjer", ratio)
        print("Product", (dict[(node, x)]+1)*factor1 * w * ratio)
        pr += (dict[(node, x)]+1)*factor1*w*ratio
        print("Final product for node", node, pr)
        dict3.update({node : pr})
        print(" ")
        suma = sum(dict3.values())
        norm = [i/suma*100 for i in dict3.values()]
        #*******************************************
    for x in list(G.predecessors(node)):   
        print("Neighbour, node: ", x, ",",  node)
        factor1 = nodedeg(node) + nodedeg(x) - 2*G.get_edge_data(x, node)['weight']
        print("Factor1", factor1)
        print("Counter", dict[(x, node)])
        print("Counter+1", dict[(x, node)]+1)
        w = G.get_edge_data(x, node)['weight']
        print("weight", w)
        print("nodedeg(node)", nodedeg(node))
        ratio = nodedeg(node)/(nodedeg(node)+nodedeg(x))
        print("Ratio", ratio)
        print("Product", (dict[(x, node)]+1)*factor1 * w * ratio)
        pr += (dict[(x, node)]+1)*factor1*w*ratio
        print("Final product for node", node, pr)
        dict3.update({node : pr})
        print(" ")
        suma = sum(dict3.values())
        norm = [i/suma*100 for i in dict3.values()]


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

print(len(norm))

#%%


b=nx.betweenness_centrality(G)
print("*betweenness*", b)   

p=nx.pagerank(G, alpha=0.9)
print("*pagerank*", p)  
