# -*- coding: utf-8 -*-
# Imports
import authSettings as aS
import sketch2Neo2 as s2n
from py2neo import Graph
import cgcn_functions

import math
import time
import pandas as pd
desired_width=500
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 100, 'display.max_rows', 100)

import numpy as np
from plotly.offline import plot
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px
import time

import streamlit as st
from PIL import Image

#https://igraph.org/python/doc/python-igraph.pdf
import igraph as ig
import louvain
import leidenalg

# Connecting to the Neo4j Database
try:
    #local
    # graph = Graph("bolt://localhost:7687", auth=(aS.graphUser, aS.graphPass))   
    # vps
    graph = Graph("bolt://31.147.204.249:7687", auth=(aS.graphUser, aS.graphPass))
except:
    pass    

# Create indexes - just the first time for running the database
# s2n.createIndexes()

# API credentials
userName=aS.userName
apiKey=aS.apiKey

# start time
startTime= time.time()

# Sidebar logo
st.sidebar.title('ConGraCNet')
# Sidebar Corpus select
cS= st.sidebar.selectbox('Select the corpus', s2n.corpusListDf['cS'])


# Sidebar Corpus data based on the selection
try:
    corpSelect=s2n.corpusSelect(cS)
    corpus=corpSelect[1]
    language=corpSelect[2]
    corpusID=corpSelect[3]
    gramRel=corpSelect[4] #initial gramRel coordination type
except:
    pass

# Sidebar Lemma pos select
try:
    lemma = st.sidebar.text_input('Select source lemma', '', key='lemma')
    pos = '-'+st.sidebar.text_input('Select source pos', '', key='pos')
    lempos = lemma+pos
except:
    'No data for this lemma+pos'
    pass


##########################################################
# Source lexeme
st.title('Source lexeme')

'##', lempos
'Corpus: ', corpSelect[0] 

assert lemma != '', st.error('Please choose a lemma')
assert pos != '-', st.error('Please choose a pos')

st.sidebar.subheader('Network parameters')
# Sidebar Measure
limit = st.sidebar.slider('Friend coocurences', 1, 100, 15) #initial limit
# Sidebar Measure
measure=st.sidebar.selectbox('Measure',('score', 'freq')) #initial measure
# Sidebar Source lexeme harvest option
initialHarvest = st.sidebar.checkbox('Initial data harvest', value=False, key='initialHarvest') # Source harvest select
# Sidebar Friends lexeme harvest option
harvestCheck = st.sidebar.checkbox('Friend harvesting option', value= False) # Friend harvest select
if harvestCheck:
    harvestSelect = 'y'
else:
    harvestSelect = 'n'

# Frequency and Relative frequency of the source lemma in corpus
def source_lemma_freq(lemma, pos, corpusID, corpus, language, gramRel):
    try:
        q='''
        match (n:Lemma{lempos:$lemma+$pos, language:$language})
        where n[$freq]<>''
        with n[$freq] as freq, n[$relFreq] as relFreq
        return freq, relFreq
        ''' 
        df = graph.run(q, lemma= lemma, pos = pos, freq = 'freq_'+corpusID, relFreq='relFreq_'+corpusID, language= language).to_data_frame()        

    except:
        s2n.corpusSketchHarvest(cS, [lemma], aS.userName, aS.apiKey, "wsketch?", pos, pos, corpus, 1, language, gramRel)
        df = graph.run(q, lemma= lemma, pos = pos, freq = 'freq_'+corpusID, language= language ).to_data_frame()
         
    return (df)
source_lema_data = source_lemma_freq(lemma, pos, corpusID, corpus, language, gramRel)
try:
    'Frequency in corpus: ', source_lema_data['freq'].values[0]
    'Relative frequency in corpus (per million): ', source_lema_data['relFreq'].values[0]
except:
    pass

# Network construction (lemma, pos, gramRel, measure, pos2)
st.write('# Friend network')

# Initial harvest (on demand)
@st.cache(show_spinner=True)
def getInitData(lemma, pos, limit):
    'Harvests the data about the lexeme with a limit'
    try:
        s2n.corpusSketchHarvest(cS, [lemma], aS.userName, aS.apiKey, "wsketch?", pos, pos, corpus, limit, language, corpSelect[4])
        result= ("Sucess")
    except:
        result ='Some problems with harvesting, but we are moving on...'
        pass
    return (result)
# Activate source lexeme initial harvest    
if initialHarvest:
    if lemma and pos:
        st.spinner("Loading initial "+ str(limit)+ "coocurences and "+ measure+ 'as measure:'+ getInitData(lemma, pos, limit))


# GramRel list - for a lemma in a corpus give a list of gramRels and count of collocation
def lemmaGramRels(lemma, pos, corpusID, language):
    try:    
        q="""MATCH (n:Lemma{lempos:$lemma+$pos, language:$language})-[r]-(m:Lemma)
        WHERE r[$count]<>""
        WITH type(r) as rels
        RETURN distinct(rels) as gramRels, count(rels) as count
        """
        df = graph.run(q, language=language, lemma=lemma, pos=pos, count='count_'+corpusID, score='score_'+corpusID).to_data_frame()
    except:
        pass
    return(df)

listGramRels= lemmaGramRels(lemma, pos, corpusID, language)   

# GramRel list option
listGramRelsCheck =st.checkbox('GramRels count list', key= 'listGramRels') # Checkbox for gramrel list
if listGramRelsCheck:
    '', listGramRels

# Lemma network profile - for a lemma in a corpus gives a collocation profile with all gramRels
def lemmaGramRelProfile(lemma, pos, corpusID, language):
    try:
        q="""MATCH (n:Lemma{lempos:$lemma+$pos, language:$language})-[r]->(m:Lemma)
        WITH n.lempos as source, r[$count] as freq, r[$score] as score, m.lempos as friend, type(r) as rels
        RETURN source, friend, rels, freq, score order by rels, freq desc
        """
        df = graph.run(q, language=language,lemma=lemma, pos=pos, count='count_'+corpusID, score='score_'+corpusID).to_data_frame()
    except:
        'No data for this lemma'
        pass
    return(df)

lemmaProfile=st.checkbox('Lemma profile', key= 'lemmaProfile')
if lemmaProfile:
    df= lemmaGramRelProfile(lemma, pos, corpusID, language)
    '', df.dropna()


minParameters=st.checkbox('Set minimal frequency and score parameters')
if minParameters:
    scoreMin=st.number_input('Minimum score', 0)
    freqMin=st.number_input('Minimum frequency', 0)


# GramRel selection (selectbox, selection based on the listGramRels df, initial is set from the corpSelect[4])
st.sidebar.subheader('Construction parameters')
gramRel= st.sidebar.selectbox('Choose grammatical relation', listGramRels['gramRels'], list(listGramRels['gramRels']).index(corpSelect[4]))


# GramReldf (function)
def lemmaByGramRel(lemma, pos, gramRel, corpusID, measure):
    #for a lemma, gramRel in a corpus get list of collocations
    try:
        q="""
        MATCH (n:Lemma{lempos:$lemma+$pos,language:$language})-[r]-(m:Lemma{language:$language})
        WHERE type(r) = $gramRel
        WITH n.lempos as source, r[$count] as freq, r[$score] as score, m.lempos as friend
        RETURN source, friend, freq, score, $gramRel
        """
        d= graph.run(q, language=language, lemma=lemma, pos=pos, gramRel=gramRel, count='count_'+corpusID, score='score_'+corpusID).to_data_frame()
        d=d.drop_duplicates().dropna().sort_values(by=[measure], ascending=False)
    except:
        'Unsuficient data for this lemma.'
    return(d)
gramReldf= lemmaByGramRel(lemma, pos, gramRel, corpusID, measure)

# Filter pos2 (selectbox from gramReldf)
pos2='-'+st.sidebar.selectbox('Filter the pos value of the friend lexeme', gramReldf['friend'].str[-1].unique() )

# FriendCoocurences - find coocurences of lemma+pos according to the gramRel with limit and measure, harvestSelect
st.cache(show_spinner=True)
def friends_df(lemma, pos, corpus, gramRel, pos2, limit, measure, harvestSelect):
    try:
        #Query based on the score ordering
        if measure=='score':
            q='''
            MATCH p=(lempos:Lemma{lempos:$lemma+$pos, language:$language})-[r]-(lemposF:Lemma{language:$language}) 
            WHERE type(r)=$gramRel AND r[$count]<>'' AND lemposF.lempos ENDS WITH $pos2
            WITH lemposF.lempos AS lemposF, r[$score] as score, r[$count] as freq 
            ORDER BY score DESC LIMIT $limit
            RETURN distinct(lemposF) AS friend, freq, score
            '''
        #Query based on the freq ordering
        if measure=='freq':
            q='''
            MATCH p=(lempos:Lemma{lempos:$lemma+$pos, language:$language})-[r]-(lemposF:Lemma{language:$language}) 
            WHERE type(r)=$gramRel AND r[$count]<>'' AND lemposF.lempos ENDS WITH $pos2
            WITH lemposF.lempos AS lemposF, r[$score] as score, r[$count] as freq 
            ORDER BY freq DESC LIMIT $limit
            RETURN distinct(lemposF) AS friend, freq, score
            '''
        df=graph.run(q, language=language, lemma=lemma, pos=pos, corpusID=corpusID, 
                        count='count_'+corpusID, score='score_'+corpusID,
                        limit=int(limit), gramRel=gramRel, pos2=pos2, measure=measure).to_data_frame()
        df=df.drop_duplicates()
        #compensate for duplicates 
        nb=len(df)
        if nb<int(limit):
            n=int(limit)-nb
            df=graph.run(q, lemma=lemma, pos=pos, pos2=pos2, language=language, corpusID=corpusID, 
                        count='count_'+corpusID, score='score_'+corpusID, 
                        limit=int(limit)+int(n)+3, gramRel=gramRel).to_data_frame()
            df=df.drop_duplicates(subset= 'friend', keep='first') # (subset= 'friend', keep='first')       
        #for low number of friends
        if len(df) <int(limit):
            #Decision whether to harvest if the number of lemma is smaller
            if harvestSelect == 'y':
                #('Not enough friends instances. Harvesting. This might take a while, depending on the limit parameter, lexical complexity and your previous searches.')            
                try:
                    s2n.corpusSketchHarvest(cS, [lemma], aS.userName, aS.apiKey, "wsketch?", pos, pos2, corpus, limit, language, gramRel)
                except:
                    pass  

        # Add gramRel, source, corpus columns
        df['gramRel']= gramRel
        df['source']= lemma+pos
        df['corpus']= corpusID
        # drop duplicates from the friend dataset (sometimes the friends have small measure numbers (bug) )
        df=df.drop_duplicates(subset= 'friend', keep='first') 
    except:
        'Unsufficient data for friend network'
        pass
    return (df)
# activate function friends_df() and store as friendCoocurences variable
friendCoocurences = friends_df(lemma, pos, corpus, gramRel, pos2, limit, measure, harvestSelect)

# List dataframe friends
'', len(friendCoocurences),' collocations of ', str(lemma+pos), ' in ',  gramRel, "ordered by", measure, " in ", corpus
'', friendCoocurences


# Draw Friends Score vs Frequency Rank
def drawFriendCoocurences(friendCoocurences): 
    'this function prints the result of a friendCoocurences(lemma, pos, corpus, limit)'
    #Draw the coocurences from friendcoocurences dataframe df
    df= friendCoocurences
    df['Rank'] = ''
    df['Rank'] = np.arange(1, len(df) + 1)
    trace1 = go.Scatter(x = df.Rank, y = df.score,
                        mode = "lines+markers+text", 
                        name = "Score",
                        marker = dict(color = 'rgba(16, 112, 2, 0.8)'), 
                        text= df.friend,
                        yaxis='y')
    # Creating trace2
    trace2 = go.Scatter(x = df.Rank, y = df.freq,
                        mode = "lines+markers",
                        name = "Frequency",
                        marker = dict(color = 'rgba(80, 26, 80, 0.5)'),
                        text= df.friend,
                        yaxis='y2')
    data = [trace1, trace2]
    layout = go.Layout(title=' Score vs log(Frequency) for '+df.gramRel[0]+' '+df.source[0], 
                        xaxis=dict(title='Rank'), yaxis=dict(title='Score'), yaxis2=dict(title='Frequency', type='log', titlefont=dict(color='rgb(148, 103, 189)'), tickfont=dict(color='rgb(148, 103, 189)'), overlaying='y', side='right'))
    fig = dict(data = data, layout = layout)
    st.plotly_chart(fig)

drawFriendCoocurences(friendCoocurences)

# Friend IGraph
def Fgraph(friendCoocurences):
    #create df variable
    df=friendCoocurences[['source', 'friend', measure]]
    #create tuples from df.values
    tuples = [tuple(x) for x in df.values]
    #create igraph object from tuples
    G=ig.Graph.TupleList(tuples, directed = False, edge_attrs=['weight'], vertex_name_attr='name', weights=False)
    #create vertex labels from name attr
    G.vs["label"]= G.vs["name"]
    G.vs["degree"]=G.vs.degree()
    G.vs["pagerank"]=G.vs.pagerank(directed=False, weights='weight')
    G.vs["personalized_pagerank"]=G.vs.personalized_pagerank(directed=False, weights='weight')
    G.vs["color"] = "rgba(255,0,0,0.2)"
    return(G)
Fgraph = Fgraph(friendCoocurences) 


# Draw Friend Igraph
st.subheader('Friend graph visualization')
def FgraphDraw(Fgraph, Glayout):
    G = Fgraph
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
    ig.plot(G, "images/Fgraph.png", **visual_style)
    return(G)

#call FgraphDraw function
FgraphDraw(Fgraph, 'fr')
#represent the Fgraph image with caption
Fgraph_image = Image.open('images/Fgraph.png')
st.image(Fgraph_image, caption='Friends graph for '+lemma+pos
    +' with vertices: '+str(Fgraph.vcount())
    +', edges: '+str(Fgraph.ecount())
    +', graph density: '+str(Fgraph.density(loops=False))
    +', diameter: '+str(Fgraph.diameter(directed=False, unconn=True, weights=None))
    ,use_column_width=True)


# Friend-of-friend network
st.title('Friend of friend network')

@st.cache(show_spinner=True)
# For a source lexeme extract friends in gramRel, and their friends
def FoFData(friendCoocurences, lemma, pos, corpus, gramRel, pos2, limit, measure, harvestSelect):
    #start with Friend dataset
    df_fof = friendCoocurences[['source', 'friend', 'freq','score','gramRel','corpus']]
    #extract friends of FoF network
    try:            
        for index, row in df_fof.iterrows():
            df2= friends_df(row['friend'][0:-2], row['friend'][-2:], corpus, gramRel, pos, limit, measure, harvestSelect)        
            #append the dataframe of df2 to df
            df2= df2.rename(columns={'friend': 'source', 'source':'friend'})
            df_fof=df_fof.append(df2, sort=False)
            df_fof=df_fof.drop_duplicates().dropna()   
    except:
        pass
    return(df_fof)

# Store Friend-of-friend network dataframe in a df_fof variable
df_fof = FoFData(friendCoocurences, lemma, pos, corpus, gramRel, pos2, limit, measure, harvestSelect)
'', df_fof #[['source','friend', 'freq', 'score', 'gramRel']]
'FoF dataset for source lexeme', lemma+pos, 'with ', len(df_fof), 'collocations'

# create FoF Igraph
@st.cache(allow_output_mutation=True)
def FoF_graph(df_fof, measure):
    #create tuples from df.values
    df= df_fof[['source', 'friend', measure]]
    tuples = [tuple(x) for x in df.values]
    #Create igraph object from tuples
    G = ig.Graph.TupleList(tuples, directed = False, edge_attrs =['weight'], vertex_name_attr='name', weights=False)
    G= G.simplify(combine_edges=max)
    #create vertices labels from name attr
    G.vs["label"]=G.vs["name"]
    G.vs["degree"]=G.vs.degree()
    G.vs['betweenness'] = G.betweenness(vertices=None, directed=False, cutoff=None, weights=None, nobigint=True)
    G.vs["pagerank"]=G.vs.pagerank(directed=False, weights='weight')
    G.vs["personalized_pagerank"]=G.vs.personalized_pagerank(directed=False, damping=0.85, reset=None, reset_vertices=None, weights='weight')
    #add weighted degree to vs: https://igraph.org/python/doc/igraph.GraphBase-class.html#strength
    G.vs["weighted_degree"] = G.strength(G.vs["label"], weights='weight', mode='ALL') #OUT|IN give the same measure on the nondirected 
    G.vs['shape']="circle"
    G.vs["color"] = "rgba(255,0,0,0.2)"
    return(G)

#store the result in a FoFgraph variable
FoFgraph=FoF_graph(df_fof, measure)

st.subheader('FoF graph visualization')


####################################################### Visualization selection
st.sidebar.subheader('Visualization parameters')

# Vertex size selection 
vertexSizeType = st.sidebar.selectbox('Vertex size by', ('weighted_degree', 'degree', 'freq'))
if vertexSizeType == 'weighted_degree':
    if measure == 'score':
        vertexSizeValues= (0.0, 2.0, 1.0)
if vertexSizeType == 'degree':
    vertexSizeValues= (0.0, 10.0, 5.0)
vertexSize= st.sidebar.slider('Vertex size', vertexSizeValues[0], vertexSizeValues[1], vertexSizeValues[2])


# Label size selection
vertexLabelSizeType = st.sidebar.selectbox('Label size by', ('weighted_degree', 'degree'))
if vertexLabelSizeType == 'weighted_degree':
    vertexLabelSizeValues= (1.0, 40.0, 10.0)
if vertexLabelSizeType == 'degree':
    vertexLabelSizeValues= (1.0, 40.0, 10.0)
vertexLabelSize= st.sidebar.slider('Label size', vertexLabelSizeValues[0], vertexLabelSizeValues[1], vertexLabelSizeValues[2])

# Layout selection
layout= st.sidebar.selectbox('Layout type', ('kk', 'fr', 'lgl'))
# FoFgraphDraw
def FoFgraphDraw(FoFgraph, layout):
    G=FoFgraph
    visual_style = {}
    visual_style["vertex_size"] = [i * vertexSize for i in G.vs[vertexSizeType]]
    visual_style["vertex_color"] = "rgba(255,0,0,0.2)"
    visual_style["edge_color"] = "rgba(255,0,0,0.2)"
    visual_style["vertex_label_color"] = "black"
    visual_style["vertex_label_size"] = [math.log(1/(1/(i+5)))*vertexLabelSize for i in G.vs[vertexLabelSizeType]]
    visual_style["vertex_label_dist"] = 0.2
    visual_style["edge_width"] = G.es["weight"]
    visual_style['hovermode'] = 'closest'
    visual_style["layout"] = layout
    visual_style["bbox"] = (1500, 1500)
    visual_style["margin"] = 90
    ig.plot(G, "images/FoFgraph.png", **visual_style)
    return (G)
# visualize network
if st.checkbox('Visualize network'):
    FoFgraphDraw(FoFgraph, layout)
    fof_image = Image.open('images/FoFgraph.png')
    st.image(fof_image, caption='Friends-of-friends graph for '+lemma+pos
        +' with vertices: '+str(FoFgraph.vcount())
        +', edges: '+str(FoFgraph.ecount())
        +', max degree: '+str(FoFgraph.maxdegree(vertices=None, loops=False))
        +', graph density: '+str(FoFgraph.density(loops=False))
        +', average_path_length: '+str(FoFgraph.average_path_length())
        +', independence: '+str(FoFgraph.alpha())
        +', diameter: '+str(FoFgraph.diameter(directed=False, unconn=True, weights=None))
        , use_column_width=True)


# Clustering Algorithms
st.title('Clusters')
st.sidebar.subheader('Clustering parameters')
# algorithm selection 
algorithm = st.sidebar.selectbox('Cluster algorithm type', ('leiden', 'louvain'))
partitionType= st.sidebar.selectbox('Partition type', ('mvp', 'cpm'))

#create function for extracting paterns
def clusterAlgo(FoFgraph, algorithm, partitionType):
    G= FoFgraph
    # Leiden
    if algorithm == 'leiden':
        #Modularity Vertex Partition
        if partitionType == 'mvp':
            partition = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition)
        #CPM Vertex Partition applies resolution_parameter (1 - every node is a community | 0- all nodes areone community)
        if partitionType == 'cpm':
            resolution = st.slider('Resolution', 0.0, 1.0, 0.5)
            partition = leidenalg.find_partition(G, leidenalg.CPMVertexPartition, resolution_parameter=resolution)
    
    # Louvain
    if algorithm == 'louvain':
        #Modularity Vertex Partition
        if partitionType == 'mvp':
            partition = louvain.find_partition(G, louvain.ModularityVertexPartition)
        #CPM Vertex Partition applies resolution_parameter (1 - every node is a community | 0- all nodes areone community)
        if partitionType == 'cpm':
            resolution = st.slider('Resolution', 0.0, 1.0, 0.5)
            partition = louvain.find_partition(G, louvain.CPMVertexPartition, resolution_parameter=resolution)
    
    return (partition)
clusterAlgo = clusterAlgo(FoFgraph, algorithm, partitionType)
'', clusterAlgo

# cluster visualization
st.subheader('Cluster visualization')

def clusterAlgoDraw(FoFgraph, clusterAlgo, layout, vertexSize, vertexLabelSize, imageName, edgeLabelSize):
    G=FoFgraph
    partition=clusterAlgo
    #Cluster Colors Programatic Method
    palette = ig.drawing.colors.ClusterColoringPalette(len(partition))
    #Vertex transparency
    vtransparency = 0.1 # vertex transparency
    G.vs['color'] = palette.get_many(partition.membership)
    vcolors=[]  
    for v in G.vs['color']:
        #convert tuples to list
        vcolor=list(v)
        #add opacity value
        vcolor[3]= vtransparency
        vcolor=tuple(vcolor)
        vcolors.append(vcolor)
    G.vs['color'] = vcolors
    #Edge color transparency
    transparency = 0.3 # edges transparency
    G.es['color']="rgba(0,0,0,0.1)"
    for p in range(len(partition)): 
        edge = (G.es.select(_within = partition[p]))
        edge['color'] = palette.get_many(p)
        tr_edge=[]
        for c in edge['color']:
            #convert tuples to list, add opacity value, reconvert to tuples
            lst=list(c)
            #set opacity value
            lst[3]= transparency
            tr_edge.append(lst) 
        tr_edge=tuple(tr_edge)
        edge['color'] = tr_edge

    #visual_style settings
    visual_style = {}
    # visual_style["vertex_size"] = [i * 0.7 for i in G.vs[vertexSizeType]]#1000, pagerank
    visual_style["vertex_size"] = [math.log(1/(0.5/(5*i+30)))*vertexSize for i in G.vs[vertexSizeType]]
    visual_style["vertex_label_color"] = "rgba(0,0,0,0.8)"
    # visual_style["vertex_label_size"] = [vertexLabelSize + math.log10(i) for i in G.vs[vertexLabelSizeType]]#30 #maybe it could be G.vs["degree"]
    visual_style["vertex_label_size"] = [math.log(1/(1/(i+5)))*vertexLabelSize for i in G.vs[vertexLabelSizeType]]
    visual_style["vertex_label_dist"] = 0
    visual_style["vertex_color"] = G.vs["color"]
    visual_style["vertex_shape"]= G.vs['shape']
    visual_style["edge_color"] = G.es['color']
    visual_style["edge_width"] = G.es["weight"]
    visual_style["edge_label"] = G.es["weight"]#could be set edgeLabelType gramRel
    visual_style["edge_label_size"] = edgeLabelSize
    visual_style['hovermode'] = 'closest'
    visual_style["layout"] = layout
    visual_style["bbox"] = (1500, 1500)
    visual_style["margin"] = 70
    visual_style["edge_curved"] = True   
    
    ig.plot(partition, "images/"+imageName+".png", **visual_style)
if st.checkbox('Visualize clusters'):
    # activate visualisation function clusterAlgoDraw
    clusterAlgoDraw(FoFgraph, clusterAlgo, layout, vertexSize, vertexLabelSize, 'FoFClusterAlgo', 2.0)
    # visualise clusters
    clusterGraph = Image.open('images/FoFClusterAlgo.png')
    st.image(clusterGraph, caption='Clustered Friends-of-friends graph'
    +' with vertices: '+str(FoFgraph.vcount())
        +', edges: '+str(FoFgraph.ecount())
        +', max degree: '+str(FoFgraph.maxdegree(vertices=None, loops=False))
        +', graph density: '+str(FoFgraph.density(loops=False))
        +', average_path_length: '+str(FoFgraph.average_path_length())
        +', independence: '+str(FoFgraph.alpha())
        +', diameter: '+str(FoFgraph.diameter(directed=False, unconn=True, weights=None))
        , use_column_width=True)

st.subheader('Community exploration')
# Create dataframe from Leiden partitions
def AlgoDataFrame(FoFgraph, clusterAlgo):
    #create a FoF graph G
    G = FoFgraph
    G.vs['betweenness'] = G.betweenness(vertices=None, directed=True, cutoff=None, weights=None, nobigint=True)
    #store graph measures
    partitions = clusterAlgo
    commsDataFrame=pd.DataFrame()
    #set community partition counter
    commPart=-1
    # loop through partitions and set properties: label, degree, pagerank, commPart
    for partition in partitions:
        try:
            commPart=commPart+1
            G.vs['commPart']=commPart
            commsDataFramePart=pd.DataFrame({'label': G.vs[partition]['label'],
                                             'degree': G.vs[partition].degree(),
                                             'pagerank': G.vs[partition].pagerank(),
                                             'commPart': G.vs[partition]['commPart'],
                                             'betweenness': G.vs[partition]['betweenness']                                   
                                             })
            #for each partition and node in partition get some properties
            commsDataFramePart['pagerankPart_mean']=sum(G.vs[partition].pagerank())/len(G.vs[partition])
            commsDataFramePart['degreePart_mean']=sum(G.vs[partition].degree())/len(G.vs[partition])
            #append partition to dataframe
            commsDataFrame= commsDataFrame.append(commsDataFramePart)
        except:
            pass
    # add pagerank mean for all nodes

    commsDataFrame['pagerankAll_mean'] = commsDataFrame['pagerank'].mean()
    # add degree mean for all nodes
    commsDataFrame['degreeAll_mean']= commsDataFrame['degree'].mean()

   

    return (commsDataFrame)

commsDataFrame=AlgoDataFrame(FoFgraph, clusterAlgo)
algo_sort_by = st.multiselect('Sort nodes by', commsDataFrame.columns)
try:    
    st.write(commsDataFrame.sort_values(by=algo_sort_by, ascending=False))
except:
    pass



# 'Community_multilevel', FoFgraph.community_multilevel(weights='weight', return_levels=False)
#####################################
#had to create a new way of drawing graph - deal with this later 

def FgraphDraw_new(Fgraph, layout, vertexSize, vertexLabelSize, imageName, edgeLabelSize):
    G = Fgraph
    visual_style = {}
    visual_style["vertex_size"] = [math.log(1/(1/(i+1)))*vertexSize for i in G.vs["betweenness"]]#vertexSize
    visual_style["vertex_label_color"] = "rgba(0,0,0,0.7)"
    visual_style["vertex_label_size"] = [math.log(1/(1/(i+10)))*vertexLabelSize for i in G.vs["degree"]] #maybe it could be G.vs["degree"]
    visual_style["vertex_color"] = G.vs["color"]#"rgba(255,0,0,0.1)"
    visual_style["vertex_shape"]= G.vs['shape']
    visual_style["vertex_label_dist"] = 0
    visual_style["edge_color"] = G.vs['color']
    visual_style["edge_width"] = G.es["weight"]
    visual_style["edge_label"] = G.es["weight"]
    visual_style["edge_label_size"] = edgeLabelSize
    visual_style['hovermode'] = 'closest'
    visual_style["layout"] = layout#G.layout_fruchterman_reingold(weights=G.es["weight"], maxiter=1000, area=len(G.es)**3, repulserad=len(G.es)**3)#Glayout
    visual_style["bbox"] = (1500, 1500)
    visual_style["margin"] = 60
    visual_style["edge_curved"] = True
    ig.plot(G, 'images/'+imageName+".png", **visual_style)
    return(G)



#################### cluster 2nd degree friends by coordination
# identify source-friends network using a grammatical relation (not coordination) and cluster friends using f_graph with relation=coordination
def f_connected_by_coordination(lemma, pos, corpus, gramRel, pos2, limit, measure, harvestSelect):
    # get friends df
    df = friends_df(lemma, pos, corpus, gramRel, pos2, limit, measure, harvestSelect)[['source', 'friend', 'freq','score','gramRel','corpus']]
    # get friend_of_friends_by_coordination
    try:            
        for index, row in df.iterrows():
            df2= friends_df(row['friend'][0:-2], row['friend'][-2:], corpus, s2n.corpusSelect(cS)[4], pos2, limit, measure, harvestSelect)        
            #append the dataframe of df2 to df
            df2= df2.rename(columns={'friend': 'source', 'source':'friend'})
            df=df.append(df2, sort=False)
            df=df.drop_duplicates().dropna()   
    except:
        pass
    
    # create a FoF graph
    G= FoF_graph(df, measure)
    G.vs["label"]= G.vs["name"]
    G.vs["degree"]=G.vs.degree()
    G.vs['betweenness'] = G.betweenness(vertices=None, directed=False, cutoff=None, weights=None, nobigint=True)
    G.vs['shape']="circle"
    G.vs["pagerank"]=G.vs.pagerank(directed=False, weights='weight')
    G.vs["color"] = "rgba(255,0,0,0.2)"
    return (G)


if not gramRel == s2n.corpusSelect(cS)[4]:
    st.title('Cluster 2nd degree friends by coordination')
    st.markdown('This part of the application is activated if 1st degree friend network is NOT created from coordination relation. It gives the friend of friend relations by applying coordination. The resulting graph should represent possibilities of other coocurences.')
    f_connected_by_coordination= f_connected_by_coordination(lemma, pos, corpus, gramRel, pos2, limit, measure, harvestSelect)  
    'f_connected_by_coordination graph'
    vertexSize=st.slider('vertexSize', 0.0,10.0,5.0)
    vertexLabelSize=st.slider('vertexLabelSize', 0.0,10.0,5.0)
    edgeLabelSize=st.slider('edgeLabelSize', 0.0,10.0,5.0)
    # FgraphDraw_new(f_connected_by_coordination, layout, vertexSize, vertexLabelSize, 'f_connected_by_coordination', edgeLabelSize)
    f_con_by_coordCluster = louvain.find_partition(f_connected_by_coordination, louvain.ModularityVertexPartition)
    clusterAlgoDraw(f_connected_by_coordination, f_con_by_coordCluster, layout, vertexSize, vertexLabelSize, 'f_connected_by_coordination', edgeLabelSize)
    fof_image = Image.open('images/f_connected_by_coordination.png')
    st.image(fof_image, caption='Friends connected by coordination graph for '+lemma+pos
        +' with vertices: '+str(FoFgraph.vcount())
        +', edges: '+str(FoFgraph.ecount())
        +', max degree: '+str(FoFgraph.maxdegree(vertices=None, loops=False))
        +', graph density: '+str(FoFgraph.density(loops=False))
        +', average_path_length: '+str(FoFgraph.average_path_length())
        +', independence: '+str(FoFgraph.alpha())
        +', diameter: '+str(FoFgraph.diameter(directed=False, unconn=True, weights=None))
        , use_column_width=True)
    
    'Clusters in graph', louvain.find_partition(f_connected_by_coordination, louvain.ModularityVertexPartition)

    

# Pruned version of FoFgraph
st.title('Pruned graph')
#betweenness parameter
btwn = FoFgraph.betweenness()#weights='weight'
ntile_prun=st.slider('Betweenness percentile', 0.1,100.0,50.0)
ntile_betweenness = np.percentile(btwn, ntile_prun)
pruned_vs = FoFgraph.vs.select([v for v, b in enumerate(btwn) if b >= ntile_betweenness])
pruned_graph = FoFgraph.subgraph(pruned_vs)
# 'pruned_graph', type(pruned_graph)
# 'pruned_graph vertices:', len(pruned_graph.vs),'edges:', len(pruned_graph.es), 'diversity', pruned_graph.diversity(), 'degree', pruned_graph.degree()

# prune by degree
pruned_graph.vs['degree']= pruned_graph.degree()
maxDegree= pruned_graph.maxdegree(vertices=None, loops=False)
prunedDegree = st.slider('Degree pruned', 0, maxDegree, (2,maxDegree))
pruned_graph_degree_vs = pruned_graph.vs.select(degree_ge = prunedDegree[0], degree_le= prunedDegree[1])
pruned_graph = FoFgraph.subgraph(pruned_graph_degree_vs)
#graph viz parameters  
vertexSize_prun=st.slider('vertexSize pruned', 0.0,10.0,5.0)
vertexLabelSize_prun=st.slider('vertexLabelSize  pruned', 0.0,10.0,5.0)
edgeLabelSize_prun=st.slider('edgeLabelSize  pruned', 0.0,10.0,5.0)
# resolution parameter
cluster_resolution_prun=st.slider('cluster resolution  pruned', 0.0,1.0,0.5)

# clusters of pruned_graph
# clusterAlgo_prun = leidenalg.find_partition(pruned_graph, leidenalg.ModularityVertexPartition) # ovo radi ali bi se trebalo moć modulirati
clusterAlgo_prun = leidenalg.find_partition(pruned_graph, leidenalg.CPMVertexPartition, resolution_parameter=cluster_resolution_prun)
'Pruned graph clusters:', clusterAlgo_prun
# prune by pruned_community
pruned_DataFrame=AlgoDataFrame(pruned_graph, clusterAlgo_prun)
if st.checkbox('View DataFrame select'):
    pruned_DataFrame


comm_pruned_select = st.multiselect('Community number', pruned_DataFrame['commPart'].unique())
if comm_pruned_select:
    pruned_DataFrame= pruned_DataFrame[pruned_DataFrame['commPart'].isin(comm_pruned_select)]
    'Selected community', pruned_DataFrame
    pruned_graph_comm_vs = pruned_graph.vs.select(label_in = pruned_DataFrame['label'].tolist())
    pruned_graph = pruned_graph.subgraph(pruned_graph_comm_vs)

### Uncomment this to see static representation
    FgraphDraw_new(pruned_graph, layout, vertexSize_prun, vertexLabelSize_prun, 'pruned', edgeLabelSize_prun)
else:
    clusterAlgoDraw(pruned_graph, clusterAlgo_prun, layout, vertexSize_prun, vertexLabelSize_prun, 'pruned', edgeLabelSize_prun)

# display pruned image
pruned_image = Image.open('images/pruned.png')
st.image(pruned_image, caption='Pruned graph for '+lemma+pos
    +' with vertices: '+str(pruned_graph.vcount())
    +', edges: '+str(pruned_graph.ecount())
    +', max degree: '+str(pruned_graph.maxdegree(vertices=None, loops=False))
    +', graph density: '+str(pruned_graph.density(loops=False))
    +', average_path_length: '+str(pruned_graph.average_path_length())
    +', independence: '+str(pruned_graph.alpha())
    +', diameter: '+str(pruned_graph.diameter(directed=False, unconn=True, weights=None))
    , use_column_width=True)

# Plotly graphing function 
plotly_Fgraph=cgcn_functions.FgraphDraW_plotly(pruned_graph, layout, vertexSize_prun, vertexLabelSize_prun, 'F_graph_plotly_pruned', edgeLabelSize_prun)
st.plotly_chart(plotly_Fgraph)

# TODO!! # Plotly graphing function with clusters  
# plotly_Fgraph_cluster=cgcn_functions.FgraphDraW_plotly_cluster(pruned_graph, clusterAlgo_prun, layout, vertexSize_prun, vertexLabelSize_prun, 'F_graph_plotly_pruned', edgeLabelSize_prun)
# st.plotly_chart(plotly_Fgraph_cluster)


##################### Find categorical structure with is-a construction
st.title('Find categorical structure with is-a construction')
st.markdown('''
Create a ntework *N_source(1,2...n) is-a N_category* to identify the more abstract category for a set of clustered words. 
It is applied on the pruned coordination clusters.
The algorithm:  

* a) **For each lemma in a *pruned cluster* identify `is-a` friends (target)**. 
The friends network should be created with *directed relations*.
We expect the more abstract *category* to appear in the is-a target node.  
* b) **Create is-a target friends network and identify the central node**. 
We expect the *category* to appear as the central node of the FoF network
''')

st.subheader('Find common target nodes method')
# extract the number of the community with the source lemma
commPart_source_pruned = pruned_DataFrame[pruned_DataFrame['label']==lemma+pos]['commPart']
# extract the clustered lemmas with the source lemma
commPart_comm_source_pruned = pruned_DataFrame[pruned_DataFrame['commPart'].isin(commPart_source_pruned)].label
# if comm_pruned_select:
#     'commPart_comm_source_pruned= ',pruned_DataFrame
'community for extraction of is-a:'
commPart_comm_source_pruned

def create_network(lemposSeries):
    df=pd.DataFrame()
    for row in lemposSeries:
        df_friend= friends_df(row.split('-')[0], '-'+row.split('-')[1], corpus, '"%w" is a ...', '-n', limit, measure, harvestSelect)
        df= df.append(df_friend)
        # except:
        #     pass
        
    return df
is_a_net= create_network(commPart_comm_source_pruned)
# 'df append', is_a_net
# 'df groupby friend, count', is_a_net.groupby(['friend']).count().sort_values(by='freq', ascending=False)['freq']
# 'df groupby score, mean', is_a_net.groupby(['friend']).mean().sort_values(by='score', ascending=False)['score']
# 'df groupby freq, mean', is_a_net.groupby(['friend']).mean().sort_values(by='freq', ascending=False)['freq']

def cluster_network(dataframe, measure):
    #create df variable
    df=dataframe[['source', 'friend', measure]]
    #create tuples from df.values
    tuples = [tuple(x) for x in df.values]
    #create igraph object from tuples
    G=ig.Graph.TupleList(tuples, directed = True, edge_attrs=['weight'], vertex_name_attr='name', weights=False)
    #create vertex labels from name attr
    G.vs["label"]= G.vs["name"]
    G.vs["degree"]=G.vs.degree()
    G.vs["pagerank"]=G.vs.pagerank(directed=True, weights='weight')
    G.vs["personalized_pagerank"]=G.vs.personalized_pagerank(directed=True, weights='weight')
    G.vs["weighted_degree"] = G.strength(G.vs["label"], weights='weight', mode='ALL')
    G.vs["color"] = "rgba(255,0,0,0.2)"
    G.vs["betweenness"] = G.betweenness()
    return(G)
is_a_net_network= cluster_network(is_a_net, measure)
is_a_slider= st.slider('Prune is-net_network  by degree', 0,10,0)
if is_a_slider>0:
    is_a_net_network_vs = is_a_net_network.vs.select(degree_gt = is_a_slider)
    is_a_net_network = is_a_net_network.subgraph(is_a_net_network_vs)
# Plotly graphing function 
is_a_net_network_Fgraph=cgcn_functions.FgraphDraW_plotly(is_a_net_network, layout, vertexSize_prun, vertexLabelSize_prun, 'is_a_graph_plotly_pruned', edgeLabelSize_prun)
st.plotly_chart(is_a_net_network_Fgraph)

# 'is_a_net network', is_a_net_network
is_a_net_df= pd.DataFrame(columns=['label', 'degree', 'pagerank', 'weighted_degree', 'betweenness'])
is_a_net_df['label']=is_a_net_network.vs['name']
is_a_net_df['degree']=is_a_net_network.vs['degree']
is_a_net_df['pagerank']=is_a_net_network.vs['pagerank']
is_a_net_df['weighted_degree']=is_a_net_network.vs['weighted_degree']
is_a_net_df['betweenness']=is_a_net_network.vs['betweenness']
'is_a_net_df', is_a_net_df.sort_values(by='pagerank', ascending=False)
# 'maxdegree',is_a_net_network.vs['weighted_degree']#select(_degree = is_a_net_network.maxdegree())["name"]


######################################
# for lemma, pos in 
def f_is_a_from_coordinationComm(lemposSeries, corpus, gramRel, pos2, limit, measure, harvestSelect):
    df= pd.DataFrame()
    for row in lemposSeries:
        lemma = row.split('-')[0]
        pos =  '-'+ row.split('-')[1]
        # get is a... friends network for a lemma
        # try:
        isA_f_df = friends_df(lemma, pos, corpus, '"%w" is a...', pos, limit, measure, harvestSelect)
        # append to df
        df = df.append(isA_f_df) 
        # except:
        #     pass
        # try:            
        #     for index, row in df.iterrows():
        #         df2= friends_df(row['friend'][0:-2], row['friend'][-2:], corpus, s2n.corpusSelect(cS)[4], pos2, limit, measure, harvestSelect)        
        #         #append the dataframe of df2 to df
        #         df2= df2.rename(columns={'friend': 'source', 'source':'friend'})
        #         df=df.append(df2, sort=False)
        #         df=df.drop_duplicates().dropna()   
        # except:
        #     pass

        # G= FoF_graph(df, measure)
        # G.vs["label"]= G.vs["name"]
        # G.vs["degree"]=G.vs.degree()
        # G.vs['betweenness'] = G.betweenness(vertices=None, directed=False, cutoff=None, weights=None, nobigint=True)
        # G.vs['shape']="circle"
        # G.vs["pagerank"]=G.vs.pagerank(directed=False, weights='weight')
        # G.vs["color"] = "rgba(255,0,0,0.2)"
        return (df)


if gramRel == s2n.corpusSelect(cS)[4]:
    'df is a', f_is_a_from_coordinationComm(commPart_comm_source_pruned, corpus, gramRel, pos2, limit, measure, harvestSelect)  
    # 'f_connected_by_coordination graph'
    # vertexSize=st.slider('vertexSize', 0.0,10.0,5.0)
    # vertexLabelSize=st.slider('vertexLabelSize', 0.0,10.0,5.0)
    # edgeLabelSize=st.slider('edgeLabelSize', 0.0,10.0,5.0)
    # # FgraphDraw_new(f_connected_by_coordination, layout, vertexSize, vertexLabelSize, 'f_connected_by_coordination', edgeLabelSize)
    # f_con_by_coordCluster = louvain.find_partition(f_connected_by_coordination, louvain.ModularityVertexPartition)
    # clusterAlgoDraw(f_connected_by_coordination, f_con_by_coordCluster, layout, vertexSize, vertexLabelSize, 'f_connected_by_coordination', edgeLabelSize)
    # fof_image = Image.open('images/f_connected_by_coordination.png')
    # st.image(fof_image, caption='Friends connected by coordination graph for '+lemma+pos
    #     +' with vertices: '+str(FoFgraph.vcount())
    #     +', edges: '+str(FoFgraph.ecount())
    #     +', max degree: '+str(FoFgraph.maxdegree(vertices=None, loops=False))
    #     +', graph density: '+str(FoFgraph.density(loops=False))
    #     +', average_path_length: '+str(FoFgraph.average_path_length())
    #     +', independence: '+str(FoFgraph.alpha())
    #     +', diameter: '+str(FoFgraph.diameter(directed=False, unconn=True, weights=None))
    #     , use_column_width=True)
    
    # 'Clusters in graph', louvain.find_partition(f_connected_by_coordination, louvain.ModularityVertexPartition)



##################### find synsets


##################### compare two graphs





################################################ find path
st.title('Path finding algorithms')
pathSource= st.text_input('Select a source lexeme from the network', lemma+pos)#, commsDataFrame['label'])
pathTarget = st.text_input('Write lempos value of a Target lexeme')
kPaths = st.number_input('How many paths?', 2)
pathDirection = st.selectbox('Select direction of the path', ('IN', 'OUT', 'BOTH'))

st.cache()
def findPath(pathSource, pathTarget, kPaths,gramRel, pathDirection, corpusID):
    q='''
    MATCH (start:Lemma{lempos:$pathSource}), (end:Lemma{lempos:$pathTarget})
    CALL algo.kShortestPaths.stream(start, end, $kPaths, $weightMeasure,
        {nodeQuery:'Lemma', relationshipQuery:'`'+$gramRel+'`', defaultValue:1.0, write:'true', direction:$pathDirection})
    YIELD index, nodeIds, costs
    RETURN [node in algo.getNodesById(nodeIds) | node.lempos] AS lempos, costs,
       reduce(acc = 0.0, cost in costs | acc + cost) AS totalCost
    '''
    df= graph.run(q, pathSource=pathSource, pathTarget=pathTarget, kPaths= kPaths, gramRel=gramRel, weightMeasure='score_'+corpusID, pathDirection=pathDirection)
    return (df)

pathResult = findPath(pathSource, pathTarget, kPaths,gramRel, pathDirection, corpusID).to_data_frame()
'K_ShortestPaths in', corpusID, ': ', pathSource, '--[',  gramRel, ']--', pathTarget
for index, row in pathResult.iterrows():
    st.write(row)

posMiddle=st.text_input('Choose your connecting POS (-n, -v, -j, -r, etc..)')
def findMiddle(pathSource, pathTarget, posMiddle, pathDirection, corpusID):
    q='''
    MATCH p= (start:Lemma{lempos:$pathSource})--> (middle:Lemma{lpos:$posMiddle})--> (end:Lemma{lempos:$pathTarget}) 
    return middle.lempos
    '''
    df= graph.run(q, pathSource=pathSource, pathTarget=pathTarget, posMiddle=posMiddle, pathDirection=pathDirection, corpusID=corpusID )
    return (df)
connecting=findMiddle(pathSource, pathTarget, posMiddle, pathDirection, corpusID)
if connecting:
    'Connecting lexemes', connecting.to_data_frame().drop_duplicates()



# EndTime
endTime= time.time()
'ConGraCNet calculations finished in', endTime-startTime

# To do
# napraviti da se samo koordinacija izračunava kao neusmjereni graf a svi ostali kao usmjereni