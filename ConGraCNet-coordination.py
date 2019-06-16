#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 12:48:45 2019
@author: Benedikt Perak
"""
#%% I Database Connect section

from py2neo import Graph
import math 

def neo4jConnect(user, password):
    #Authentication 
    return (Graph("bolt://localhost:7687", auth=(user, password)))

user=input("what is your username? ")
password=input("what is your password? ")

#set the graph variable for activating py2neo graph functions
graph = neo4jConnect(user, password)


# II Function section

# Set the parameters for the construction of the Friend networks
#set the limit of coocurence instances for and_orF
limit=15 #max = 300
#set the minimum coocurence score for and_orF
scoreMin=0 #max=14
#set the minimum coocurence count/freq for and_orF
freqMin=0

#%% Functions for extracting the F network and FoF network
# =============================================================================
#to do - bug -> distinct 

#extract Source-Friend coocurence network with and/or type of syntactic-semantic relation
#ententen13 "%w" and/or ... 
#hrwac koordinacija

def and_orF(lemma, pos, corpus):
    #creates a Friend of a Friend network from the DataBase
    if corpus == "ententen13":
        q='''
        MATCH p=(n:Lemma{lempos:$lemma+$pos})-[r:`"%w" and/or ...`]-(m:Lemma) 
        WHERE m.lempos ENDS WITH $pos AND r.score_ententen13_tt2_1>$scoreMin AND r.count_ententen13_tt2_1>$freqMin
        RETURN distinct n.name as source, m.name as friend, r.score_ententen13_tt2_1 as weight 
            
        order by weight desc limit $limit
        //r.count_ententen13_tt2_1 as freq 
        '''
    if corpus == "hrwac":
        q='''
        MATCH p=(n:Lemma{lempos:$lemma+$pos})-[r:`koordinacija`]-(m:Lemma) 
        WHERE m.lempos ENDS WITH $pos AND r.score_hrwac22>$scoreMin AND r.count_hrwac22>$freqMin
        RETURN distinct n.name as source, m.name as friend, r.score_hrwac22 as weight 
        order by weight desc limit $limit 
        '''
    
    if corpus == "europarl7":
        q='''
        MATCH p=(n:Lemma{lempos:$lemma+$pos})-[r:`and/or`]-(m:Lemma) 
        WHERE m.lempos ENDS WITH $pos AND r.score_europarl7_en>$scoreMin AND r.count_europarl7_en>$freqMin
        RETURN distinct n.name as source, m.name as friend, r.score_europarl7_en as weight 
        order by weight desc limit $limit 
        '''

    if corpus == "europarl_plenary":
        q='''
        MATCH p=(n:Lemma{lempos:$lemma+$pos})-[r:`"%w" and/or ...`]-(m:Lemma) 
        WHERE m.lempos ENDS WITH $pos AND r.score_europarl_plenary>$scoreMin AND r.count_europarl_plenary>$freqMin
        RETURN distinct n.name as source, m.name as friend, r.score_europarl_plenary as weight 
        order by weight desc limit $limit
        '''
        
    if corpus == "croparl":
        q='''
        MATCH p=(n:Lemmas{lempos:$lemma+$pos})-[r:`Lemmas_conj`]-(m:Lemmas) 
        WHERE m.lempos ENDS WITH $pos AND r.freqInCorp>$freqMin
        with distinct m, n, r
        with distinct(n.lempos) as source, m.lempos as friend, log(r.freqInCorp) as weight 
        with split(source, $pos) as source, split(friend, $pos) as friend, weight
        return source[0] as source, friend[0] as friend, weight
        order by weight desc limit $limit 

        '''

    
    df=graph.run(q, lemma=lemma, pos=pos, scoreMin=scoreMin, freqMin=freqMin, limit=limit).to_data_frame()
    df2=df.query('friend == source')
    
    #concatenate df2 with df and drop these duplicates'
    df=pd.concat([df,df2]).drop_duplicates(keep=False)
    
    #require len(df)=limit
    if len(df)<limit:
        n=limit-len(df)
        df=graph.run(q, lemma=lemma, pos=pos, scoreMin=scoreMin, freqMin=freqMin, limit=limit+n).to_data_frame()
        df3=df.query('friend == source')
        df=pd.concat([df3,df]).drop_duplicates(keep=False)
    
    return (df)


#Parameters a
lemma="future"
pos="-n"
corpus="europarl7" # "croparl", "hrwac", "ententen13", "europarl7"
limit=30

#and_orF(lemma, pos, corpus)
#%% extract FoF coocurence network with "%w" and/or ... type of syntactic-semantic relation

### popraviti da ne radi u oba smjera
def and_orFoF(lemma, pos, corpus):
    #extract F network
    df=and_orF(lemma, pos, corpus)
    
    #extract FoF network from friends 
    for index, row in df.iterrows():
        df2=and_orF(row['friend'], pos, corpus)
        #append the dataframe of df2 to df
        df=df.append(df2)
    
    #drop duplicate rows
    df=df.drop_duplicates()
    
    #drop rows with friend==source
    #find in dataframe rows where friend==source'
    df3=df.query('friend == source')
    #concatenate df3 with df and drop these duplicates'
    df=pd.concat([df,df3]).drop_duplicates(keep=False)
    
    return (df)

limit=20
and_orFoF("mačka", "-n", "hrwac")
#and_orFoF("krava", "-n", "croparl")
#and_orFoF("vote", "-v", "ententen13")  


#%%#Function list Nodes (Sources, FriendsAll, Friends)

def nodesFoF(lemma, pos, corpus):
    df =  and_orFoF(lemma, pos, corpus)  
    #Lexemes that are sources
    sources= df.source.unique()
    #Friend lexemes that did not become Source
    f=df.query('friend not in source')
    friends=f.friend.unique()
    
    return(sources, len(sources), friends, len(friends))

limit=20
nodesFoF("trava", "-n", "hrwac")
#%% 3 Function for importing and_orFoF dataframe as a tuple list and creating a Igraph object
#https://igraph.org/python/doc/python-igraph.pdf
import igraph as ig
#create friend Graph
def Fgraph(lemma, pos, corpus, Glayout):
    
    #create df variable
    df=and_orF(lemma, pos, corpus)
    #create tuples from df.values
    tuples = [tuple(x) for x in df.values]
    #create igraph object from tuples
    G=ig.Graph.TupleList(tuples, directed = False, edge_attrs=['weight'], vertex_name_attr='name', weights=False)
    #create vertex labels from name attr
    G.vs["label"]=G.vs["name"]
    G.vs["degree"]=G.vs.degree()
    G.vs["pagerank"]=G.vs.pagerank(directed=False, weights='weight')
    print(G.vs["pagerank"])
    G.vs["personalized_pagerank"]=G.vs.personalized_pagerank(directed=False, weights='weight')
    visual_style = {}
    visual_style["vertex_size"] = [i * 2 for i in G.vs["degree"]]
    visual_style["vertex_label_color"] = "black"
    visual_style["vertex_label_size"] = 50 #[i *10 for i in G.vs["pagerank"]] #maybe it could be G.vs["degree"]
    visual_style["vertex_color"] = "rgba(255,0,0,0.2)"
    visual_style["edge_color"] = "rgba(255,0,0,0.2)"
    visual_style["vertex_label_dist"] = 1
    visual_style["edge_width"] = G.es["weight"]
    visual_style["edge_label"] = G.es["weight"]
    visual_style['hovermode'] = 'closest'
    visual_style["layout"] = Glayout
    visual_style["bbox"] = (1500, 1500)
    visual_style["margin"] = 150
    
    lemmaW= (bytes(lemma, 'utf-8')).decode('mbcs', 'ignore')
    print(G)
    ig.plot(G, "images/Fgraph_"+lemmaW+pos+"-"+str(limit)+"-"+str(scoreMin)+"-"+str(freqMin)+"-"+corpus+"-"+Glayout+".png", **visual_style)
    ig.plot(G, "images/Fgraph_"+lemmaW+pos+"-"+str(limit)+"-"+str(scoreMin)+"-"+str(freqMin)+"-"+corpus+"-"+Glayout+".svg", **visual_style)

limit=4
#Fgraph("strah", "-n", "hrwac", "fr")
#Fgraph("krava", "-n", "croparl", "fr")
#Fgraph("chair", "-n", "ententen13", "fr")
#Fgraph("fear", "-n", "eurparl7")
#%% create FoF graph
def FoFgraph(lemma, pos, corpus):
    #create df variable
    df=and_orFoF(lemma, pos, corpus)
    #create tuples from df.values
    tuples = [tuple(x) for x in df.values]
    #Create igraph object from tuples
    G=ig.Graph.TupleList(tuples, directed= False, edge_attrs =['weight'], vertex_name_attr='name', weights=False)
    G.simplify(combine_edges=max)
    #create vertices labels from name attr
    G.vs["label"]=G.vs["name"]
    G.vs["degree"]=G.vs.degree()
    G.vs["pagerank"]=G.vs.pagerank(directed=False, weights='weight')
    G.vs["personalized_pagerank"]=G.vs.personalized_pagerank(directed=False, damping=0.85, reset=None, reset_vertices=None, weights='weight')
    #add weighted degree to vs: https://igraph.org/python/doc/igraph.GraphBase-class.html#strength
    G.vs["weighted_degree"] = G.strength(G.vs["label"], weights='weight', mode='ALL')#OUT|IN give the same measure on the nondirected 
    #print(G.vs['degree'], G.vs['name'], G.es['weight'])
    #print (G.vs["label"],G.vs["weighted_degree"])
    return(G)

limit=4
FoFgraph("osjećaj", "-n", "hrwac")
#FoFgraph("krava", "-n", "croparl")
#FoFgraph("chair", "-n", "ententen13")

#%% FoFgraphDraw
def FoFgraphDraw(lemma, pos, corpus, Glayout):
    G=FoFgraph(lemma, pos, corpus)
    G.vs["pagerank"]=G.vs.pagerank(directed=False, weights='weight')
    visual_style = {}
    visual_style["vertex_size"] = [i * 1 for i in G.vs["weighted_degree"]] #5 "degree",
    visual_style["vertex_color"] = "rgba(255,0,0,0.2)"
    visual_style["edge_color"] = "rgba(255,0,0,0.2)"
    visual_style["vertex_label_color"] = "black"
    visual_style["vertex_label_size"] = 40 #maybe it could be G.vs["degree"]
    visual_style["vertex_label_dist"] = 0.2
    visual_style["edge_width"] = G.es["weight"]
    visual_style['hovermode'] = 'closest'
    visual_style["layout"] = Glayout
    visual_style["bbox"] = (1500, 1500)
    visual_style["margin"] = 90
    
    lemmaW= (bytes(lemma, 'utf-8')).decode('mbcs', 'ignore')
    
    ig.plot(G, "images/FoFgraph_"+lemmaW+pos+"-"+corpus+"-"+str(limit)+"-"+str(scoreMin)+"-"+str(freqMin)+"-"+Glayout+".png", **visual_style)
    ig.plot(G, "images/FoFgraph_"+lemmaW+pos+"-"+corpus+"-"+str(limit)+"-"+str(scoreMin)+"-"+str(freqMin)+"-"+Glayout+".svg", **visual_style)
    return (G)

limit=10
#FoFgraphDraw("krava", "-n", "croparl", "fr")
#FoFgraphDraw("stolica", "-n", "hrwac", "fr")
#FoFgraphDraw("love", "-n", "ententen13", "fr")
#%% 4 process the graph with louvain
import louvain

#create function for extracting paterns
def louvainAlg(lemma, pos, corpus):
    
    G= FoFgraph(lemma, pos, corpus)
    partition = louvain.find_partition(G, louvain.CPMVertexPartition, resolution_parameter = 0.011)
    print(partition)

limit=15
scoreMin=0 #max=14
#set the minimum coocurence count/freq for and_orF
freqMin=0


#louvainAlg("krava", "-n", "croparl")
#louvainAlg("osjećaj", "-n", "hrwac")
louvainAlg("chair", "-n", "ententen13")
#louvainAlg("fisheries", "-n", "europarl_plenary")


#%% 4 Process the graph with leidenalg
# =============================================================================
import leidenalg

#create function for extracting paterns
def leidenAlg(lemma, pos, corpus):
    G= FoFgraph(lemma, pos, corpus)
    partition = leidenalg.find_partition(G, leidenalg.CPMVertexPartition, resolution_parameter = 0.0105)
    print(partition)

limit=15
scoreMin=0 #max=14
#set the minimum coocurence count/freq for and_orF
freqMin=0


#leidenAlg("krava", "-n", "croparl")
#leidenAlg("osjećaj", "-n", "hrwac")
leidenAlg("chair", "-n", "ententen13")
#leidenAlg("fisheries", "-n", "europarl_plenary")
#%% function for extracting communities and drawing the object
#todo napraviti node frequency

def leidenAlgoDraw(lemma, pos, corpus, Glayout, partitionType, resolution):
    
    '''Creates a leiden Partition from a graph object + a layout'''
    G= FoFgraph(lemma, pos, corpus)
    if partitionType == 'mvp':
        partition = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition)
    if partitionType == 'cpm':
        partition = leidenalg.find_partition(G, leidenalg.CPMVertexPartition, resolution_parameter=resolution)
    
    #Cluster Colors Programatic Method
    palette = ig.drawing.colors.ClusterColoringPalette(len(partition))
    
    
    #Vertex color
    G.vs['color'] = palette.get_many(partition.membership)
    
    #Edge color
    for p in range(len(partition)): 
        edge = (G.es.select(_within = partition[p]))
        
        edge['color'] = palette.get_many(p)
        
        
        for c in edge['color']:
            #convert tuples to list, add opacity value, reconvert to tuples
            lst=list(c)
            #set opacity value
            lst[3]= 0.1
            lst2 = tuple(lst)
            print(lst2)
    
                  
            
    #visual_style settings
    #layout=G.layout(Glayout)
    visual_style = {}
    visual_style["vertex_size"] = [i * 0.7 for i in G.vs["weighted_degree"]]#1000, pagerank
    visual_style["vertex_label_color"] = "black"
    visual_style["vertex_label_size"] = 30 #maybe it could be G.vs["degree"]
    visual_style["vertex_label_dist"] = 0
    visual_style["edge_color"] = G.es['color']
    visual_style["edge_width"] = G.es["weight"]
    visual_style['hovermode'] = 'closest'
    visual_style["layout"] = Glayout
    visual_style["bbox"] = (1500, 1500)
    visual_style["margin"] = 70
    print(partition)
    print(G.summary())
    
    lemmaW= (bytes(lemma, 'utf-8')).decode('mbcs', 'ignore')
    
    ig.plot(partition, "images/FoFLeiden_"+lemmaW+pos+"_"+corpus+"-"+str(limit)+"-"+str(scoreMin)+"-"+str(freqMin)+"-"+Glayout+".svg", **visual_style)
    ig.plot(partition, "images/FoFLeiden_"+lemmaW+pos+"_"+corpus+"-"+str(limit)+"-"+str(scoreMin)+"-"+str(freqMin)+"-"+Glayout+".png", **visual_style)



#set the range of the graph
limit=15 #max = 300
#set the minimum coocurence score for and_orF
scoreMin=0 #max=14
#set the minimum coocurence count/freq for and_orF
freqMin=0
#mvp,cpm 

#leidenAlgoDraw("religija", "-n", "hrwac", "fr", "mvp", 0.05)
#leidenAlgoDraw("krava", "-n", "croparl", "fr",  "mvp", 0.05)
leidenAlgoDraw("emotion", "-n", "ententen13", "fr", "mvp", 0.09) 
#leidenAlgoDraw("Nice", "-n", "europarl7", "fr", "cpm", 0.5) #loaded in different graph
#leidenAlgoDraw("culture", "-n", "europarl_plenary", "fr", "mvp", 0.02)
#%% function for optimising patterns
def leidenAlgOpti(lemma, pos, corpus):
    #create FoF graph
    G = FoFgraph(lemma, pos, corpus)
    #find partitions
    partition = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition)
    #create optimiser
    print(len(partition))
    optimiser= leidenalg.Optimiser().optimise_partition(partition, n_iterations=10)
    #print the value of the optimiser
    print(optimiser)
    partition2 = leidenalg.find_partition(G, leidenalg.CPMVertexPartition, resolution_parameter = optimiser)
    print(len(partition2))

leidenAlgOpti("chair", "-n", "ententen13")
#%% resolution_parameter
limit=5
resolution = 0.02
#resolution_parameter=
#create FoF graph

G = FoFgraph("chair", "-n", "ententen13")
partition = leidenalg.find_partition(G, leidenalg.CPMVertexPartition, resolution_parameter=resolution)
print(len(partition))
print(partition)

#%% Function with CPMVertex resolution control
def make_partition(lemma, pos, corpus, res):
        G = FoFgraph(lemma, pos, corpus)
        partition = leidenalg.find_partition(G, leidenalg.CPMVertexPartition, resolution_parameter=res)
        #print(partition)
        print(len(partition))
        return partition

limit=13
print(make_partition("chair", "-n", "ententen13", 0.02))

#%% Function: Get back desired number of communities
def desired_len_partition(lemma, pos, corpus, res, increment, partitionDesired):
    partitions= make_partition(lemma,pos,corpus, res)
    number= len(partitions)
    
    while number > partitionDesired:
        final_partition= make_partition(lemma,pos,corpus, res)
        number = len(final_partition)
        res -= increment
    
    while number < partitionDesired:
        final_partition= make_partition(lemma,pos,corpus, res)
        number = len(final_partition)
        res += increment

    
    print(final_partition)
    print(res)
    return (final_partition)    
        
    
limit= 13
desired_len_partition("chair", "-n", "ententen13", 0.5, 0.01, 7)

#%%
def draw_desired_len_partition(lemma, pos, corpus, res, increment, partitionDesired, Glayout):
    
    G = FoFgraph(lemma, pos, corpus)
    partition = leidenalg.find_partition(G, leidenalg.CPMVertexPartition, resolution_parameter=res)
    
    #Cluster Colors Programatic Method
    palette = ig.drawing.colors.ClusterColoringPalette(len(partition))
    #Vertex color
    G.vs['color'] = palette.get_many(partition.membership)
    #Edge color
    for p in range(len(partition)): 
        edge = (G.es.select(_within = partition[p]))
        edge['color'] = palette.get_many(p)
        for c in edge['color']:
            #convert tuples to list, add opacity value, reconvert to tuples
            lst=list(c)
            #set opacity value
            lst[3]= 0.3
            lst = tuple(lst)
            #print(lst)
            
            
            
    #visual_style settings
    #layout=G.layout(Glayout)
    visual_style = {}
    visual_style["vertex_size"] = [i * 1000 for i in G.vs["pagerank"]]
    visual_style["vertex_label_color"] = "black"
    visual_style["vertex_label_size"] = 30 #maybe it could be G.vs["degree"]
    visual_style["vertex_label_dist"] = 1
    visual_style["edge_color"] = G.es[lst]
    visual_style["edge_width"] = G.es["weight"]
    visual_style['hovermode'] = 'closest'
    visual_style["layout"] = Glayout
    visual_style["bbox"] = (1500, 1500)
    visual_style["margin"] = 70
    print(partition)
    
    lemmaW= (bytes(lemma, 'utf-8')).decode('mbcs', 'ignore')
    
    ig.plot(partition, "images/FoFLeiden_Desired_"+lemmaW+pos+"_"+corpus+"-"+str(limit)+"-"+str(scoreMin)+"-"+str(freqMin)+"-"+partitionDesired+"-"+Glayout+".svg", **visual_style)
    ig.plot(partition, "images/FoFLeiden_Desired_"+lemmaW+pos+"_"+corpus+"-"+str(limit)+"-"+str(scoreMin)+"-"+str(freqMin)+"-"+partitionDesired+"-"+Glayout+".png", **visual_style)

limit= 7
draw_desired_len_partition("chair", "-n", "ententen13", 0.5, 0.01, 2, "fr")
#%% optimiser 1:profile 
limit=20
G = FoFgraph("bitch", "-n", "ententen13")
partition = leidenalg.find_partition(G, leidenalg.CPMVertexPartition, resolution_parameter=0.1)
print(len(partition))

profile= leidenalg.Optimiser().resolution_profile(G,leidenalg.CPMVertexPartition, resolution_range=(0,1))

#%% optimiser 2: optimiser
diff=2
while diff>1:
    diff = leidenalg.Optimiser().optimise_partition(partition, n_iterations=10)
#print the value of the optimiser
    print(diff)
    partition2 = leidenalg.find_partition(G, leidenalg.CPMVertexPartition, resolution_parameter = diff)
    print(len(partition2))




# =============================================================================
# CroParl
# =============================================================================

#%% Zastupnik uses lexeme
def zast_lexeme(lemma, pos, corpus):
    if corpus == "croparl":
        if pos == "-n":
            upostag ="NOUN"
        q='''
        MATCH p = (z:Zastupnik)-[di:DELIVERED_izjavu]->(i:Izjava)-[hs:HAS_sentence]->(s:Sentences)-[ht:HAS_token]->(t:Tokens{lemma:$lemma, upostag:$upostag}) 
        RETURN z.name as source, $lemma as friend, log(count(ht)) as weight order by weight desc limit 15
        '''
    df=graph.run(q, lemma=lemma, upostag=upostag).to_data_frame()
    return (df)

zast_lexeme("krava", "-n", "croparl")

#%% Zastupnik frequency of lexeme and example
import pprint

def zast_lexeme_ex(lemma, pos, corpus):
    if corpus == "croparl":
        if pos == "-n":
            upostag ="NOUN"
        q='''
        MATCH p = (z:Zastupnik)-[di:DELIVERED_izjavu]->(i:Izjava)-[hs:HAS_sentence]->(s:Sentences)-[ht:HAS_token]->(t:Tokens{lemma:$lemma, upostag:$upostag})        
        with z.name as Person, $lemma as friend, count(ht) as FrequencyOfUse, s.text as SentenceExample order by FrequencyOfUse desc limit 15
        return Person, SentenceExample, FrequencyOfUse
        '''
    df=graph.run(q, lemma=lemma, upostag=upostag)#.to_data_frame()
    #print(df['Person'],['FrequencyOfUse']) 
    pp = pprint.PrettyPrinter(indent=3)
    pp.pprint(list(df))
    
    #return (df)

zast_lexeme_ex("osjećaj", "-n", "croparl")


#%% lexeme and_orF df + Zastupnik uses F
def zast_lexeme_and_orF(lemma, pos, corpus):
    #extract F network
    df=and_orF(lemma, pos, corpus)
    #extract FoF network from friend 
    for index, row in df.iterrows():
        df2=zast_lexeme(row['friend'], pos, corpus)
        #append the dataframe of df2 to df
        df=df.append(df2)
    df=df.drop_duplicates()
    return (df)



zast_lexeme_and_orF("ustaša", "-n", "croparl")
#%% draw F graph with Zastupnik
import time

#create friend Graph
def zastupnik_Fgraph(lemma, pos, corpus, Glayout):
    start = time.time()
    #create df variable
    df=zast_lexeme_and_orF(lemma, pos, corpus)
    #create tuples from df.values
    tuples = [tuple(x) for x in df.values]
    #create igraph object from tuples
    G=ig.Graph.TupleList(tuples, directed = False, edge_attrs=['weight'], vertex_name_attr='name', weights=False)
    partition = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition)
    
    #create vertex labels from name attr
    G.vs["label"]=G.vs["name"]
    G.vs["degree"]=G.vs.degree()
    G.vs["pagerank"]=G.vs.pagerank()
    G.vs["personalized_pagerank"]=G.vs.personalized_pagerank()
    
    #Cluster Colors Programatic Method
    palette = ig.drawing.colors.ClusterColoringPalette(len(partition))
    #Vertex color
    G.vs['color'] = palette.get_many(partition.membership)
    #Edge color
    for p in range(len(partition)): 
        edge = (G.es.select(_within = partition[p]))
        edge['color'] = palette.get_many(p)
        for c in edge['color']:
            #convert tuples to list, add opacity value, reconvert to tuples
            lst=list(c)
            #set opacity value
            lst[3]= 0.1
            lst = tuple(lst)
            
            
    #visual_style settings
    #layout=G.layout(Glayout)
    visual_style = {}
    visual_style["vertex_size"] = [i * 1000 for i in G.vs["pagerank"]]
    visual_style["vertex_label_color"] = "black"
    visual_style["vertex_label_size"] = 30 #maybe it could be G.vs["degree"]
    visual_style["vertex_label_dist"] = 1
    visual_style["edge_width"] = G.es["weight"]
    visual_style['hovermode'] = 'closest'
    visual_style["layout"] = Glayout
    visual_style["bbox"] = (1500, 1500)
    visual_style["margin"] = 150
    print(partition)
    
    lemmaW= (bytes(lemma, 'utf-8')).decode('mbcs', 'ignore')

    ig.plot(partition, "images/Fgraph_Zast_"+lemmaW+pos+"-"+str(limit)+"-"+str(scoreMin)+"-"+str(freqMin)+"-"+corpus+"-"+Glayout+".png", **visual_style)
    ig.plot(partition, "images/Fgraph_Zast_"+lemmaW+pos+"-"+str(limit)+"-"+str(scoreMin)+"-"+str(freqMin)+"-"+corpus+"-"+Glayout+".svg", **visual_style)
    end = time.time()
    print("Graph created in: ", end-start)

limit=10
zastupnik_Fgraph("fašizam", "-n", "croparl", "fr")


#%% III Query: Extract the coocuring nodes (Friends)
friends=and_orF("audience", "-n", "ententen13")
print(friends)

#%% III Query+Draw Friend network (lemma, pos, layout)
Fgraph("stolica", "-n", "hrwac", "fr")


#%% III Query: Extract the graph and sort it according to the 
g=FoFgraph("chair", "-n", "ententen13")
for item in sorted(g.vs, key= lambda x:x['degree']):
    print (item["name"], item["degree"])#, (item["pagerank"]*1000), (item["personalized_pagerank"]*1000) )

#%% III Query: Find partitions according to the 
#leidenAlg(lemma, pos)
leidenAlg("chair", "-n")

#%%% III Query: Optimiser value
leidenAlgOpti("chair", "-n")

#%% III Query and draw: Find partitions according to the leidenAlg and draw graph 
#leidenAlgoDraw(lemma, pos, igraphlayout)
leidenAlgoDraw("stolica", "-n", "hrwac", "kk")



#%% work area


#%%edge selection
#https://igraph.org/python/doc/igraph.EdgeSeq-class.html
#http://cneurocvs.rmki.kfki.hu/igraph/screenshots2.html


G = FoFgraph(lemma, pos, corpus)
#diameter=G.get_diameter

#find partitions
partition = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition)
partitionNo= len(partition)
prva= G.es.select(_within = partition[0])
druga= G.es.select(_within = partition[1]) 
#color = G.vs(partition[0][0]) ["vertex_color"]
prva["color"] =  "red"
druga["color"] = "green"
ig.plot(partition, layout="fr")

#%% Ovaj način crtanja radi do nekog broja


G = FoFgraph("chair", pos, corpus)
#diameter=G.get_diameter

G.es["color"] = "black"

#find partitions
#partition = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition)
partition = leidenalg.find_partition(G, leidenalg.CPMVertexPartition, resolution_parameter = 0.02)

#odredi boje
listC=["red", "green", "blue", "yellow", "violet", "lila", "black", "cyan", "darkgreen", "orange", "magenta"]
#listC=[1,2,3,4,5,6,7,8,9,10,11,12]

if len(partition) > len(listC):
    print ("error")

# za svaku particiju iscrtaj veze drugom bojom 
for p in range(len(partition)): 
    edge = (G.es.select(_within = partition[p]))
    color = listC[p]
    edge["color"]= listC[p]
ig.plot(partition, "images/Work.png", opacity= 0.5, layout="kk")

#%% opći princip s modulom igraph.drawing.colors
G = ig.Graph.Barabasi(n = 20, m = 1)
i = G.community_infomap()
palette = ig.drawing.colors.ClusterColoringPalette(len(i))
G.vs['color'] = palette.get_many(i.membership)
for p in range(len(i)): 
    edge = (g.es.select(_within = i[p]))
    edge['color'] = palette.get_many(p)
    #get(p)
    

#g.es['color'] = pal.get_many(i.membership) #ne printa kako treba
ig.plot(g)
#%% draw procedure for Leiden
lemma="chair"
pos="-n"
igraphlayout="fr"

G = FoFgraph(lemma, pos, corpus)
#diameter=G.get_diameter

#find partitions
#partition = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition)
partition = leidenalg.find_partition(G, leidenalg.CPMVertexPartition, resolution_parameter = 0.09)

#Cluster Colors Programatic Method
palette = ig.drawing.colors.ClusterColoringPalette(len(partition))
#Vertex color
G.vs['color'] = palette.get_many(partition.membership)
#Edge color
for p in range(len(partition)): 
    edge = (G.es.select(_within = partition[p]))
    edge['color'] = palette.get_many(p)

ig.plot(partition, "images/Work"+lemma+pos+"-"+str(limit)+"-"+str(scoreMin)+"-"+str(freqMin)+"-"+igraphlayout+".png", opacity= 1, layout="fr")
ig.plot(partition, "images/Work"+lemma+pos+"-"+str(limit)+"-"+str(scoreMin)+"-"+str(freqMin)+"-"+igraphlayout+".svg", opacity= 1, layout="fr")


#%% print the nodes in parition
for n in range(len(partition)):
    print(len(G.es.select(_within = partition[n])))
    
#%% Full draw procedure for clustering and coloring
#layout="fr"
limit=50 #max = 300
lemma= "chair"
pos="-n"

G = FoFgraph(lemma, pos, corpus)
#print(G.es['weight'])

partition = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition)


#Cluster Colors Programatic Method
palette = ig.drawing.colors.ClusterColoringPalette(len(partition))
#Vertex color
G.vs['color'] = palette.get_many(partition.membership)
#Edge color
for p in range(len(partition)): 
    edge = (G.es.select(_within = partition[p]))
    edge['color'] = palette.get_many(p)
    
        
#visual_style settings
layout="fr"
visual_style = {}
visual_style["vertex_size"] = [i * 5000 for i in G.vs["pagerank"]]
#visual_style["vertex_size"] = 10
#visual_style["vertex_color"] = "red" #ako želimo bez bojica čvorove odkomentirati
#visual_style["label"] = G.vs["name"]
visual_style["vertex_label_color"] = "black"
visual_style["vertex_label_size"] = 80 #[i * 50 for i in G.vs["pagerank"]]
visual_style["vertex_label_dist"] = 1

visual_style["edge_width"] = G.es["weight"]
#visual_style["edge_color"] = "black" #ako želimo bez bojica veze odkomentirati

visual_style['hovermode'] = 'closest'
visual_style["layout"] = layout
visual_style["bbox"] = (5500, 5500)
visual_style["margin"] = 50


ig.plot(partition , "images/FoF"+lemma+pos+"-"+str(limit)+"-"+str(scoreMin)+"-"+str(freqMin)+"-"+layout+".svg", **visual_style)
ig.plot(partition, "images/FoF"+lemma+pos+"-"+str(limit)+"-"+str(scoreMin)+"-"+str(freqMin)+"-"+layout+".png", opacity= 1, **visual_style )
#%%

#fig = ig.Plot()
#fig.add(G, layout="fr", opacity=0.25, vertex_label=None)
#fig.add(edges, layout="fr", edge_color="red", vertex_label=None)
#fig.show()

#%% Optimiser
G = FoFgraph("sound", "-n")
partition = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition)
print(partition)

#Optimiser
optimiser= leidenalg.Optimiser().optimise_partition(partition, n_iterations=40)
#optimized= optimiser.optimise_partition(partition, n_iterations=10)
print(optimiser)

partition2 = leidenalg.find_partition(G, leidenalg.CPMVertexPartition, resolution_parameter = optimiser)
print(partition2)
ig.plot(partition2, "images/optimiser.png", layout="fr")
print(len(partition2))
