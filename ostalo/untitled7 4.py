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

#nx.add_cycle(G, [10,15,16])

G.add_edge(1, 2)
G.add_edge(2, 3)
G.add_edge(3, 4)
G.add_edge(3, 8)
G.add_edge(4, 6)
G.add_edge(5, 3)
G.add_edge(5, 7)
G.add_edge(6, 10)
G.add_edge(6, 11)
G.add_edge(6, 12)
G.add_edge(6, 13)
G.add_edge(8, 14)

#nx.add_cycle(G, [10,20,30])

#e = (1, 10)

#G.add_edge(1, 10)
#G.add_edge(4, 10)
#G.add_edge(1, 5)
#G.add_edge(1, 6)
#G.add_edge(2, 7)
#G.add_edge(4, 8)
#G.add_edge(8, 30)

# G.add_edge(1, 2, weight=0.86)
# G.add_edge(1, 3, weight=1.78)
# G.add_edge(1, 17, weight=0.6)
# G.add_edge(1, 18 , weight=0.46)
# G.add_edge(1, 19 , weight=0.56)
# G.add_edge(1, 20 , weight=0.52)
# G.add_edge(1, 21, weight=0.6)
# G.add_edge(1, 22 , weight=0.6)
# G.add_edge(1, 23 , weight=0.6)
# G.add_edge(2, 27, weight=0.6)
# G.add_edge(3, 4, weight=0.6)
# G.add_edge(3, 24,  weight=0.6)
# G.add_edge(3, 26, weight=0.6)
# G.add_edge(4, 25, weight=0.6)
# G.add_edge(4, 5, weight=2.2)
# G.add_edge(5, 6, weight=2.3)
# G.add_edge(6, 7, weight=1.7)
# #G.add_edge(6, 8, weight=0.6)
# G.add_edge(6, 9, weight=0.96)
# G.add_edge(6, 10, weight=1.1)
# G.add_edge(8, 11, weight=0.6)
# G.add_edge(8, 12, weight=0.6)
# G.add_edge(8, 13, weight=0.6)
# G.add_edge(9, 14, weight=0.86)
# G.add_edge(10, 15, weight=0.6)
# G.add_edge(10, 16, weight=0.6)


nx.draw(G, with_labels = True)



def sbb2_importance(G):
    'take graph and get the importance of the nodes as a list of float values'
    #g=G.to_networkx() 
    #st.write(G)
    ciklus = nx.cycle_basis(G.to_undirected()) # Will contain: { node_value => sum_of_degrees }
    degrees = {}
    
    dict1 = {}
    for veza in (nx.edges(G)):
        #counter_veze=0
        degrees[veza[0]] = degrees.get(veza[0], 0) + G.degree(veza[1])
        degrees[veza[1]] = degrees.get(veza[1], 0) + G.degree(veza[0])
        #print('brid i stupnjevi, resp:', veza, ',' , G.degree[veza[0]], ',' , G.degree[veza[1]])
        #print('degree[veza[0]]', G.degree[veza[0]])
        #print('degree[veza[1]]', G.degree[veza[1]])

        for cicl in ciklus:
            in_path = lambda e, path: (e[0], e[1]) in path or (e[1], e[0]) in path
            cycle_to_path = lambda path: list(zip(path+path[:1], path[1:] + path[:1]))
            in_a_cycle = lambda e, cycle: in_path(e, cycle_to_path(cycle))
            in_any_cycle = lambda e, g: any(in_a_cycle(e, c) for c in nx.cycle_basis(g))
            #print(cicl)

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
            
                  #testaaa= node, susjed
                  #print((node, susjed))
                  #print(dict1[(node, susjed)])
                  #print(dict1)
                  #print(node, G.degree(node), susjed, G.degree(susjed), counter_veze)
            #u = (G.degree(node)+dict1[(node, susjed)])*(G.degree(susjed)+dict1[(node,susjed)])
            u = (G.degree(node)+dict1[(node, susjed)]+G.degree(susjed))
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
        
        #dictionary(veza, counter_veze)
    #list1.append(veza)
    #list2.append(counter_veze)
    #print(list1)
    #print(list2)
    # dict1["Sanda"] = "Arsen"
    
    # print(dict1[(1,2)])
    # print(dict1[(1,6)])
    # print(dict1["Sanda"])

#print(sbb2_importance(G))

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
        for susjed in gp.neighborhood(G, node):
            #if node == veza[0]:
            #print(node, g.degree(node), susjed, g.degree(susjed), dict1[(node, susjed)])
            u = (G.degree(node)-dict1[(node, susjed)]-1)*(G.degree(susjed)-dict1[(node,susjed)]-1)
            w = (dict1[(node, susjed)]/2+1)
            z = (G.degree(node)-1)/(G.degree(node)+G.degree(susjed)-2)
            rezultat0 = u/w
            rezultat = u/w*z
            print('u, w, z, rezultat0, rezultat:', u, w, z, rezultat0, rezultat)
            suma = suma + rezultat
        print(suma + G.degree(node))
        njihova_vaznost.append(suma + G.degree(node))
        print(nx.nodes(G))
    return njihova_vaznost

print(shishi_importance(G))
        
# nasa_vaznost = []
# for node in nx.nodes(G):
#     print(node, gp.neighborhood(G, node))
#     #if node == veza[0]:
#     suma = 0.
#     for susjed in gp.neighborhood(G, node):
#         if node == veza[0]:
#             print(node, G.degree(node), susjed, G.degree(susjed), counter_veze)
#             u = (G.degree(node)+counter_veze)*(G.degree(susjed)+counter_veze)
#             w = (counter_veze/2+1)
#             z = (G.degree(node)-1.)/(G.degree(node)+G.degree(susjed)-2.)
#             rezultat0 = u*w
#             rezultat = u*w*z
#         print('u, w, z, rezultat0, rezultat:', u, w, z, rezultat0, rezultat)
#         suma = suma + rezultat
#     print(suma + G.degree(node))
#     nasa_vaznost.append(suma + G.degree(node))
# return nasa_vaznost     

#print(sbb2_importance(G))
        
        #counter = []
    
        # veze = []
        # for edge in nx.edges(G):
        #     veze.append(edge)
        # print(veze)
    
        # counters = []
        # for edge in nx.edges(G):
        #     counters.append(counter_veze)
        # print(counters)
        
        
        # list1 = [veza]
        # print(list1)
        # dictionary = {veza: counter_veze}
        # print(dictionary)
        
#         squares = []
# >>> for i in range(10):
# ...     squares.append(i * i)
    
        
    # for node in nx.nodes(G):
    #     print(node, gp.neighborhood(G, node))
        
    #     for susjed in gp.neighborhood(G, node):
    #         print(susjed)
            #print(dictionary[0])
            #if node==dictionary[0][0]:
                #print('counter_veze', counter_veze)
            #print("prvi cvor brida", veza[0])
            #print("drugi cvor brida", veza[1])
        #baza = []
        #baza = [set(veza), counter_veze]
        #print(baza)
        
       # k = 0
        #for susjed in gp.neighborhood(G, veza[0]):
            #print('vrh', veza[0])
            #print(susjed)#(), counter_veze)
        
    
            
            #print('u1, k1', u1, k1, susjed, counter_veze)
            #print('forufor', (G.degree(veza[0])-1)*(G.degree(susjed)-1), counter_veze)
            #k = k + k1
        #print('suma', k)
        
        #for node in nx.nodes(G):
            #print(node, gp.neighborhood(G, node))
            #for veza in nx.edges(G):
                #print((G.degree(veza[0])-counter_veze)*(G.degree(veza[1])-counter_veze))
                #print('GV_counter', veza, counter_veze)
            
# =============================================================================
#                 for j in range(0, len(susjed)):
#                     print((G.degree(node)-counter_veze)*(G.degree(j)-counter_veze))
# =============================================================================
                
            
        # u=(G.degree(veza[0])-counter_veze-1)*(G.degree(veza[1])-counter_veze-1)
        # u1=(G.degree(veza[0])+counter_veze)*(G.degree(veza[1])+counter_veze)
        # #print('u=', u, counter_veze)
        # lam = counter_veze/2+1
        # #print('lambda=', lam)
        # I=u/lam
        # I1=u1*lam
        # L1=I1*(G.degree(veza[0])-1)/(G.degree(veza[0])+G.degree(veza[1])-2)
        # L2=I1*(G.degree(veza[1])-1)/(G.degree(veza[0])+G.degree(veza[1])-2)
        # #print("l1",L1,"l2", L2)
        
# =============================================================================
#         for veza in nx.edges(G):
#             print(veza[0], gp.neighborhood(G, veza[0]))
#             for susjed in gp.neighborhood(G, veza[0]):
#                 print('forufor', (G.degree(veza[0])-1)*(G.degree(susjed)-1), counter_veze)
# =============================================================================
            
        
        # test = (G.degree(veza[0])-1)/(G.degree(veza[0])+G.degree(veza[1])-2)
        # #print('I=', I)
        # w=I*((G.degree(veza[0])-1)/(G.degree(veza[0])+G.degree(veza[1])-2))
        # #print('u=', u,'w=', w, )
        # my_input.append([veza, u, lam, I, w, I1, I1*((G.degree(veza[0])-1)/(G.degree(veza[0])+G.degree(veza[1])-2)),L1,L2])
        # #print("My_input: ",my_input)
        # #print('brid, u, lambda, I, w, u1*i, I1', my_input)
        
    #for element in my_input:
        #print(my_input)
        
        # brojac=0
        # for element in my_input:
            
        #     for node in nx.nodes(G):
                
        #         if node == element[0][0]:
        #             print(node, element[7])
        #             brojac = brojac+ element[7]
        #     print(brojac)
        # brojac = 0        #break
        # for node in nx.nodes(G):
        #      #print(node, gp.neighborhood(G, node))
             
        #      for susjed in gp.neighborhood(G, node):
        #          #print((node, susjed))
        #          #print(my_input)
        #          #for element in my_input:
        #          if (node, susjed)==my_input[0][0]:
        #              brojac = brojac+my_input[0][7]
        #              print('aaa', node, my_input[0][7])
        
        #print(my_input[0][0][0])
        
        
        
    #for node in nx.nodes(G):
        # for node in nx.nodes(G):
        #     for element in my_input:
        #         if node==my_input[0][0]:
        #             print(node, element[0])
            # brojac=0
            # while node==my_input[0][0][0]:
            #     print(node, my_input[0][0][0])#, my_input[node][7])
        
# =============================================================================
#         for node in nx.nodes(G):
#             print(node, gp.neighborhood(G, node))
#             for susjed in gp.neighborhood(G, node):
#                 print('lalala', G.degree(node), G.degree(susjed), counter_veze)
# =============================================================================
        
        # vrh=[]
        # brojac=0
        # for element in my_input:
        #     vrh=[element[0][0]]
        #     for cvor in nx.nodes(G): 
        #        # print (element[0][0], cvor)
        #         if element[0][0]==cvor:
        #            # print(element[0][0] ,"sedmi ****", element[7])
        #            brojac=brojac+element[7]
        #    # print("suma",brojac) 
        # #print("vrh", vrh)
        
        # #print (my_input)
        
        # for element in my_input:
        #     print (element)
            
        # for node in nx.nodes(G):
        #     if node==my_input[0][0]:
        #         print('test', node)
        

                
        
        # zbroj_stupnjeva=0
        # for i in nx.nodes(G):
        #     zbroj_stupnjeva+=G.degree(i)
        # #print('zbroj_stupnjeva', zbroj_stupnjeva)
        
        # china_vaznost = []
        # for node in nx.nodes(G):
        #     weightlast=0
        #     weightlast1=0
        #     #print(node)
        #     j=0
        #     while j < len(my_input):
        #         a=my_input[j][0]
        #         #print ("Gledamo brid", a)
        #         if node in a:
        #             #print ("proslo")
        #             weightlast+=my_input[j][4]
        #             weightlast1+=my_input[j][6]
        #             #print(weightlast1)
        #         j=j+1
                
        #for node in nx.nodes(G):
            #print(gp.neighborhood(G, node))
            #for susjed in gp.neighborhood(G, node):
                #print(G.degree(node), G.degree(susjed), counter_veze)
                #for j in range(0, len(susjed)):
                 #   print((G.degree(node)-counter_veze)*(G.degree(j)-counter_veze))
                
            #print('weight', weightlast+G.degree(node))
            #print('weight Tajana', weightlast1+G.degree(node))
            #print('weighted degree noda', *G.vs[j]["weighted_degree"])
            #china_vaznost.append(weightlast1+G.degree(node))
            
        #print('brid, u, lambda, I, w', my_input) 
        #print('zadnje', weightlast1*((G.degree(veza[0])-1)/(G.degree(veza[0])+G.degree(veza[1])-2)))

        #for neighbor in G.neighbors(node):
            #print('testst', node, neighbor)        

#print (n for n in G.neighbors(1))
       # print(my_input) 
       # print('MAde in China', china_vaznost)
        
    #print ('Broj elemenata arraya ',len(my_input))
    
    #n for n in G.neighbors(0)
    
    #print (n for n in G.neighbors(1))
  
    #for j in nx.nodes(G):

        
        #print (G)
        #dict=G[j]
        #print('Vrh ', j, 'ima susjeda', G[j] )
        #print (dict.keys())
        #k=len(dict.keys())
        #print (k)
        #for k in dict.values():print(dict[k])
    
    #zbroj_stupnjeva=0
    #for i in nx.nodes(G):
        #zbroj_stupnjeva+=G.degree(i)
    #print('zbroj_stupnjeva', zbroj_stupnjeva)
    #brojac1=0
    # brojac1 = 0
    # for node in nx.nodes(G):
    #     print(node)
        
    #     for susjed in gp.neighborhood(G, node):
    #         print (susjed)
    #         for record in my_input:
    #             if record[0][0]==node:
    #                 brojac1 = brojac1 + record[7]
    #         print (brojac1)
    #     print ("Novo vrh-> sumaL1", node, brojac1)
    #             #print(' ', susjed)
                
    # #return "That's all folks"
#  return china_vaznost
    #vaznost_sbb2_=[(round((G.degree(j)+degrees.get(j))/zbroj_stupnjeva, 5)) for j in nx.nodes(G)]
    # sbb_importance = []
    # for j in nx.nodes(g):
    #     sbb_importance.append((G.vs[j]["name"], round((g.degree(j)+degrees.get(j))/zbroj_stupnjeva, 5)))
    #     # st.write("Vaznost cvora {} je: ".format(G.vs[j]["name"]), round((g.degree(j)+degrees.get(j))/zbroj_stupnjeva, 5))
    # 'sbb_importance', sbb_importance
    #return vaznost_sbb2_




# def shishi_importance(G):
#     'take graph and get the importance of the nodes as a list of float values'
#     #g=G.to_networkx() 
#     #st.write(g)
#     ciklus = nx.cycle_basis(G.to_undirected()) # Will contain: { node_value => sum_of_degrees }
#     degrees = {}
    
#     for veza in (nx.edges(G)):
#         counter_veze=0
#         degrees[veza[0]] = degrees.get(veza[0], 0) + G.degree(veza[1])
#         degrees[veza[1]] = degrees.get(veza[1], 0) + G.degree(veza[0])
#         print('brid:', veza)
#         print('degree[veza[0]]', G.degree[veza[0]])
#         print('degree[veza[1]]', G.degree[veza[1]])

#         for cicl in ciklus:
#             in_path = lambda e, path: (e[0], e[1]) in path or (e[1], e[0]) in path
#             cycle_to_path = lambda path: list(zip(path+path[:1], path[1:] + path[:1]))
#             in_a_cycle = lambda e, cycle: in_path(e, cycle_to_path(cycle))
#             in_any_cycle = lambda e, g: any(in_a_cycle(e, c) for c in nx.cycle_basis(g))

#     my_input = [] 
#     for veza in G.edges():
#         counter_veze=0
#         if in_any_cycle(veza, G)==True:
#             #counter_veze=0
#             for cicl in (ciklus):
#                 if len(cicl)==3:
#                     c=set(cicl)
#                     set1=set(veza)
#                     set2=set(c)
#                     is_subset = set1.issubset(set2)
#                     if is_subset==True:
#                         counter_veze+=1
#                 #else:
#                     #counter_veze=0
#         print(veza, counter_veze)
            
#         u=(G.degree(veza[0])-counter_veze-1)*(G.degree(veza[1])-counter_veze-1)
#         print('u=', u)
#         lam = counter_veze/2+1
#         print('lambda=', lam)
#         I=u/lam
#         print('I=', I)
#         w=I*((G.degree(veza[0])-1)/(G.degree(veza[0])+G.degree(veza[1])-2))
#         print('w=', w)
#         my_input.append([set(veza), u, lam, I, w]) 
        
#         #zbroj_stupnjeva=0
#         #for i in nx.nodes(g):
#             #zbroj_stupnjeva+=g.degree(i)
#         #print('zbroj_stupnjeva', zbroj_stupnjeva)
        
#         china_vaznost1 = []
#         for node in nx.nodes(G):
#             weightlast=0
#             print(node)
#             j=0
#             while j < len(my_input):
#                 a=my_input[j][0]
#                 #print ("Gledamo brid", a)
#                 if node in a:
#                     #print ("proslo")
#                     weightlast+=my_input[j][4]
#                 j=j+1
#             print('weight', weightlast+G.degree(node))
#             china_vaznost1.append(weightlast+G.degree(node))
            

                
#         #print(my_input) 
#         #print('MAde in China', china_vaznost)
#     return china_vaznost1

# print(sbb2_importance(G))
# #print(shishi_importance(G))