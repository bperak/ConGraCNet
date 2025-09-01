from plotly.validators.layout import annotation
import streamlit as st
import authSettings as aS
import sketch2Neo2 as s2n
from py2neo import Graph
try:
    #local
    # graph = Graph("bolt://localhost:7687", auth=(aS.graphUser, aS.graphPass))   
    # vps
    graph = Graph("bolt://31.147.204.249:7687", auth=(aS.graphUser, aS.graphPass))
except:
    pass    

from plotly.offline import plot
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px
import numpy as np
import pandas as pd
import math
import igraph as ig
import louvain
import leidenalg


########################
# Frequency and Relative frequency of the source lemma in corpus
@st.cache()
def source_lemma_freq(lemma, pos, corpusID, corpus, language, gramRel):
    'find frequency of the source lema, if not present harvest from the s2n function'
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


# GramRel list - for a lemma in a corpus give a list of gramRels and count of collocation
@st.cache()
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


# GramReldf (function)
@st.cache()
def lemmaByGramRel(language, lemma, pos, gramRel, corpusID, measure):
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



# FriendCoocurences dataframe
st.cache(show_spinner=True)
def friends_df(language, lemma, pos, corpus, corpusID, gramRel, pos2, limit, measure, harvestSelect, direction):
    'find coocurences of lemma+pos according to the gramRel with limit and measure, harvestSelect, set and or as undirected else directed'
    if direction =='undirected':
        # Query based on the score ordering
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
    
    if direction == 'directed':        
        #Query based on the score ordering
        if measure=='score':
            q='''
            MATCH p=(lempos:Lemma{lempos:$lemma+$pos, language:$language})-[r]->(lemposF:Lemma{language:$language}) 
            WHERE type(r)=$gramRel AND r[$count]<>'' AND lemposF.lempos ENDS WITH $pos2
            WITH lemposF.lempos AS lemposF, r[$score] as score, r[$count] as freq 
            ORDER BY score DESC LIMIT $limit
            RETURN distinct(lemposF) AS friend, freq, score
            '''
        #Query based on the freq ordering
        if measure=='freq':
            q='''
            MATCH p=(lempos:Lemma{lempos:$lemma+$pos, language:$language})-[r]->(lemposF:Lemma{language:$language}) 
            WHERE type(r)=$gramRel AND r[$count]<>'' AND lemposF.lempos ENDS WITH $pos2
            WITH lemposF.lempos AS lemposF, r[$score] as score, r[$count] as freq 
            ORDER BY freq DESC LIMIT $limit
            RETURN distinct(lemposF) AS friend, freq, score
            '''
    
    if direction == 'directed2':        
        #Query based on the score ordering
        if measure=='score':
            q='''
            MATCH p=(lempos:Lemma{lempos:$lemma+$pos, language:$language})<-[r]-(lemposF:Lemma{language:$language}) 
            WHERE type(r)=$gramRel AND r[$count]<>'' AND lemposF.lempos ENDS WITH $pos2
            WITH lemposF.lempos AS lemposF, r[$score] as score, r[$count] as freq 
            ORDER BY score DESC LIMIT $limit
            RETURN distinct(lemposF) AS friend, freq, score
            '''
        #Query based on the freq ordering
        if measure=='freq':
            q='''
            MATCH p=(lempos:Lemma{lempos:$lemma+$pos, language:$language})<-[r]-(lemposF:Lemma{language:$language}) 
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
        if direction=='directed':
            df=df.drop_duplicates(subset= 'friend', keep='first')        
    #for low number of friends
    if len(df) <int(limit):
        #Decision whether to harvest if the number of lemma is smaller
        if harvestSelect == 'y':
            # Harvesting. This might take a while, depending on the limit parameter, lexical complexity and previous searches            
            try:
                s2n.corpusSketchHarvest(cS, [lemma], aS.userName, aS.apiKey, "wsketch?", pos, pos2, corpus, limit, language, gramRel)
            except:
                pass  
        # drop duplicates from the friend dataset (sometimes the friends have small measure numbers (bug) )
        if direction=='directed':
            df=df.drop_duplicates(subset= 'friend', keep='first')
    # Add gramRel, source, corpus columns
    df['gramRel']= gramRel
    df['source']= lemma+pos
    df['corpus']= corpusID 
    return (df)


@st.cache()
def is_network(lemposSeries, language, corpus, corpusID, is_gramRel, pos2, limit, measure, harvestSelect, direction):
    df=pd.DataFrame()
    for row in lemposSeries:
        try:
            is_friend= friends_df(language, row.split('-')[0], '-'+row.split('-')[1], corpus, corpusID, '"%w" is a ...', '-n', limit, measure, harvestSelect, 'directed2')
            df= df.append(is_friend)
        except:
            pass        
    return df


# Lemma network profile - for a lemma in a corpus gives a collocation profile with all gramRels
@st.cache()
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



# Draw Friends Score vs Frequency Rank
# @st.cache(suppress_st_warning=True)
def drawFriendCoocurences(friendCoocurences): 
    'plot line of a friendCoocurences with lemma names on Score & Frequency'
    df= friendCoocurences
    df['Rank'] = ''
    df['Rank'] = np.arange(1, len(df) + 1)
    fig= go.Figure(go.Scatter(x = df.Rank, y = df.score,
                        mode = "lines+markers", #+text 
                        name = "Score",
                        marker = dict(color = 'rgba(16, 112, 2, 0.5)'), 
                        text= df.friend, 
                        yaxis='y'))
    fig.add_trace(go.Scatter(x = df.Rank, y = df.freq,
                        mode = "lines+markers",
                        name = "Frequency",
                        marker = dict(color = 'rgba(80, 26, 80, 0.3)'),
                        text= df.friend, 
                        yaxis='y2'))
    fig.update_layout(go.Layout(title='Rank vs Score|log(Frequency)',#: ('+df.source[0]+')-['+df.gramRel[0]+']-(f)', 
                        margin=dict(l=10, r=10, t=20, b=20), 
                        font_size=10, font_family='Arial',
                        showlegend=False, autosize=True, hovermode='closest',         
                        xaxis=dict(title='Rank', linecolor="rgba(80, 26, 80, 0.05)"), 
                        yaxis=dict(title='Score', linecolor="rgba(16, 112, 2, 0.1)"), yaxis2=dict(title='Frequency', type='log', titlefont=dict(color='rgb(148, 103, 189)'), tickfont=dict(color='rgb(148, 103, 189)'), overlaying='y', side='right'),
                        paper_bgcolor = 'rgba(0,0,0,0)', plot_bgcolor = 'rgba(255,255,250,0.5)',
                        height=350),
                        annotations=[go.layout.Annotation(x=int(row['Rank']), y=float(row.score),
                        xref="x", yref="y",
                        text= str(row.friend),
                        align='left',
                        showarrow=False,
                        yanchor='bottom',
                        textangle=-75) for index, row in df.iterrows()]
                        )
    st.plotly_chart(fig, use_container_width=True)

# For a source lexeme extract friends in gramRel, and their friends
@st.cache(show_spinner=True)
def FoFData(language, friendCoocurences, lemma, pos, corpus, corpusID, gramRel, pos2, limit, measure, harvestSelect, direction):
    #start with Friend dataset
    df_fof = friendCoocurences[['source', 'friend', 'freq','score','gramRel','corpus']]
    #extract friends of FoF network
    try:            
        for index, row in df_fof.iterrows():
            if direction=='undirected':
                df2= friends_df(language, row['friend'][0:-2], row['friend'][-2:], corpus, corpusID, gramRel, pos, limit, measure, harvestSelect, direction)
            if direction=='directed':
                df2= friends_df(language, row['friend'][0:-2], row['friend'][-2:], corpus, corpusID, gramRel, pos, limit, measure, harvestSelect, 'directed2')        
            #append the dataframe of df2 to df
            df2= df2.rename(columns={'friend': 'source', 'source':'friend'})
            df_fof=df_fof.append(df2, sort=False)
            df_fof=df_fof.drop_duplicates().dropna()   
    except:
        pass
    return(df_fof)



# Friend IGraph
@st.cache(allow_output_mutation=True)
def Fgraph(dataset, measure):
    'calculate a graph from dataset of source - friend coocurences with a chosen measure'
    #create df variable
    df=dataset[['source', 'friend', measure]]
    #create tuples from df.values
    tuples = [tuple(x) for x in df.values]
    #create igraph object from tuples
    G=ig.Graph.TupleList(tuples, directed = False, edge_attrs=['weight'], vertex_name_attr='name', weights=False)
    #create vertex labels from name attr
    G.vs["label"]= G.vs["name"]
    G.vs["degree"]=G.vs.degree()
    G.vs["pagerank"]=G.vs.pagerank(directed=False, weights='weight')
    G.vs["personalized_pagerank"]=G.vs.personalized_pagerank(directed=False, weights='weight')
    G.vs["weighted_degree"] = G.strength(G.vs["label"], weights='weight', mode='ALL')
    G.vs["betweenness"] = G.betweenness()
    G.vs['eigen_c'] = G.eigenvector_centrality(directed=True, scale=True, weights= 'weight', return_eigenvalue=False)
    G.vs["color"] = "rgba(255,0,0,0.2)"
    return(G)

#@st.cache()
def df_from_graph(G):
    'create a dataframe from graph and sort lempos'
    df= pd.DataFrame(columns=['label', 'degree', 'pagerank', 'weighted_degree', 'betweenness', 'eigen_c'])
    try:
        df['label']=G.vs['name']
        df['degree']=G.vs['degree']
        df['pagerank']=G.vs['pagerank']
        df['weighted_degree']=G.vs['weighted_degree']
        df['betweenness']=G.vs['betweenness']
        df['eigen_c']=G.vs['eigen_c']
    except:
        pass
    return (df)

# FoF Igraph
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


@st.cache()
def FgraphDraw(Fgraph, Glayout):
    'draws a graph from a Fgraph graph and stores it as a Fgraph'
    G = Fgraph
    visual_style = {}
    visual_style["vertex_size"] = [i * 2 for i in G.vs["degree"]]
    visual_style["vertex_label_color"] = "black"
    visual_style["vertex_label_size"] = 50 #[i *10 for i in G.vs["pagerank"]] #maybe it could be G.vs["degree"]
    visual_style["vertex_color"] = "rgba(16, 112, 2, 0.2)"
    visual_style["edge_color"] = "rgba(16, 112, 2, 0.2)"
    visual_style["vertex_label_dist"] = 1
    visual_style["edge_width"] = G.es["weight"]
    visual_style["edge_label"] = G.es["weight"]
    visual_style['hovermode'] = 'closest'
    visual_style["layout"] = Glayout
    visual_style["bbox"] = (1500, 1500)
    visual_style["margin"] = 150 
    visual_style["edge_curved"] = True  
    ig.plot(G, "images/Fgraph.png", **visual_style)
    return(G)

# FoFgraphDraw
@st.cache()
def FoFgraphDraw(FoFgraph, layout, vertexSize, vertexSizeType, vertexLabelSize, vertexLabelSizeType):
    G=FoFgraph
    visual_style = {}
    visual_style["vertex_size"] = [math.log(1/(0.5/(5*i+30)))*vertexSize for i in G.vs[vertexSizeType]]#i * vertexSize for i in G.vs[vertexSizeType]
    visual_style["vertex_color"] = "rgba(16, 112, 2, 0.2)"
    visual_style["edge_color"] = "rgba(16, 112, 2, 0.2)"
    visual_style["vertex_label_color"] = "black"
    visual_style["vertex_label_size"] = [math.log(1/(1/(i+5)))*vertexLabelSize for i in G.vs[vertexLabelSizeType]]
    visual_style["vertex_label_dist"] = 0.2
    visual_style["edge_width"] = G.es["weight"]
    visual_style['hovermode'] = 'closest'
    visual_style["layout"] = layout
    visual_style["bbox"] = (1500, 1500)
    visual_style["margin"] = 90
    visual_style["edge_curved"] = True
    ig.plot(G, "images/FoFgraph.png", **visual_style)
    return (G)


# Clustering algorithm
@st.cache(allow_output_mutation=True)
def clusterAlgo(FoFgraph, algorithm, partitionType, resolution):
    'find clusters for a graph using algorithm, partitionType, resolution'
    # Define graph
    G= FoFgraph
    # Leiden
    if algorithm == 'leiden':
        #Modularity Vertex Partition
        if partitionType == 'mvp':
            partition = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition)
        #CPM Vertex Partition applies resolution_parameter (1 - every node is a community | 0- all nodes areone community)
        if partitionType == 'cpm':
            partition = leidenalg.find_partition(G, leidenalg.CPMVertexPartition, resolution_parameter=resolution)
    
    # Louvain
    if algorithm == 'louvain':
        #Modularity Vertex Partition
        if partitionType == 'mvp':
            partition = louvain.find_partition(G, louvain.ModularityVertexPartition)
        #CPM Vertex Partition applies resolution_parameter (1 - every node is a community | 0- all nodes areone community)
        if partitionType == 'cpm':
            partition = louvain.find_partition(G, louvain.CPMVertexPartition, resolution_parameter=resolution)
    return (partition)


# Create dataframe from partitions
#@st.cache(allow_output_mutation=True)
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



# Create json structure for the 3d representation
def json_FoF_clustered(df_fof, commsDataFrame, measure):
    df_nodes = commsDataFrame[['label', 'commPart']].rename(columns={'label':'id', 'commPart':'group'}).to_json(orient="records")
    df_links = pd.DataFrame(df_fof[['source','friend', measure]]).reset_index().drop(columns=['index']).rename(columns={'source':'source', 'friend':'target', measure:'value'}).to_json(orient="records")
    df='{"nodes":'+df_nodes+',"links":'+df_links+'}'
    return df#, df_links


# Draw clusters using Igraph
def clusterAlgoDraw(FoFgraph, clusterAlgo, layout, vertexSize, vertexSizeType, vertexLabelSize, vertexLabelSizeType, edgeLabelSize, imageName):
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
    G.es['color']="rgba(16, 112, 2, 0.1)"
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



################################ First degree network plot
@st.cache(allow_output_mutation=True)
def FgraphDraW_plotly(Fgraph, layout, vertexSize, vertexLabelSize, imageName, edgeLabelSize):   
    # vertex and edges atributes of Fgraph are treated as lists 
    G=Fgraph
    labels=G.vs['label']
    betweenness=G.vs['betweenness']
    degree = G.vs['degree']
    # Define edge sizes
    edgeSize = 2
    N=len(labels)
    E=[e.tuple for e in G.es]# list of edges
    layt=G.layout(layout)

    ## node x, y components of the network
    Xn=[layt[k][0] for k in range(N)]
    Yn=[layt[k][1] for k in range(N)]
    ## edges from the layout
    Xe=[]
    Ye=[]
    for e in E:
        Xe+=[layt[e[0]][0],layt[e[1]][0], None]
        Ye+=[layt[e[0]][1],layt[e[1]][1], None]
    ## Figure
    fig=go.Figure(go.Scatter(x=Xe, y=Ye,
                mode='lines', name='Edges',
                line= dict(color='rgba(16, 112, 2, 0.5)', width=edgeSize, shape='spline', smoothing=1.3),
                hoverinfo='text'
                ))
    fig.add_trace(go.Scatter(x=Xn, y=Yn,
                mode='markers+text', 
                name='Lemma',
                marker=dict(symbol='circle-dot', 
                            size= [math.log(1/(1/(i+4)))*vertexSize for i in betweenness],
                            color=G.vs['color'],
                            line=dict(color='rgba(16, 112, 2, 0.5)', 
                            width=0.5)
                                ),
                text=labels, 
                textfont=dict(
                            family="arial",
                            size=[math.log(1/(1/(i+5)))*vertexLabelSize*0.6 for i in degree],
                            color='rgba(0, 0, 0, 0.8)'#"crimson"
                        ),
                #https://chart-studio.plotly.com/~empet/15366/customdata-for-a-few-plotly-chart-types/#/
                customdata= np.stack((G.vs['betweenness'], G.vs['degree']), axis=-1),
                hovertemplate ="<b>%{text}</b><br>"+\
                                "Betweenness: %{customdata[0]}<br>"+\
                                "Degree: %{customdata[1]}"
                                )
                    )
    fig.update_layout(go.Layout(title='First degree Network', 
                showlegend=False, autosize=True, hovermode='closest',
                font_size=10, font_family='Arial', 
                height=350,
                margin=dict(l=10, r=10, t=20, b=20),
                paper_bgcolor = 'rgba(0,0,0,0)', plot_bgcolor = 'rgba(255,255,250,0.5)',
                annotations=[dict(showarrow=False, xref='paper', yref='paper', 
                xanchor='center', yanchor='top', font=dict(size=14))], 
                xaxis=dict(showline=False, zeroline=False, showgrid=False, showticklabels=False),
                yaxis=dict(showline=False, zeroline=False, showgrid=False, showticklabels=False),
                ))   
    return (fig)



####################### Clustered Second degree network graph
#@st.cache(allow_output_mutation=False)
def FgraphDraW_plotly_cluster(graph, clusterGraph, layout, vertexSize, vertexLabelSize, imageName, edgeLabelSize):
    # vertex and edges atributes of Fgraph are treated as lists 
    G=graph
    labels=G.vs['label']
    betweenness=G.vs['betweenness']
    degree = G.vs['degree']
    partition=clusterGraph
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

    # Define edge sizes
    edgeSize =0.1
    N=len(labels)
    E=[e.tuple for e in G.es]# list of edges
    layt=G.layout(layout)
    Ncolor= G.vs['color']

    ## node x, y components of the network
    Xn=[layt[k][0] for k in range(N)]
    Yn=[layt[k][1] for k in range(N)]
    ## edges from the layout
    Xe=[]
    Ye=[]
    for e in E:
        Xe+=[layt[e[0]][0],layt[e[1]][0], None]
        Ye+=[layt[e[0]][1],layt[e[1]][1], None]
    ## Figure
    fig=go.Figure(go.Scatter(x=Xe, y=Ye,
                mode='lines', name='Edges',
                line= dict(color='grey', 
                width=edgeSize, 
                shape='spline', 
                smoothing=1.3),
                hoverinfo='none'
                ))
    fig.add_trace(go.Scatter(x=Xn, y=Yn,
                mode='markers+text', name='Lemma',
                marker=dict(symbol='circle-dot', 
                            size= [math.log(1/(1/(i+4)))*vertexSize for i in betweenness],
                            color=G.vs['color'],
                            line=dict(color=G.vs['color'], 
                            width=0.5)
                                ),
                text=labels, 
                textfont=dict(
                            family="sans serif",
                            size=[math.log(1/(1/(i+5)))*vertexLabelSize*0.6 for i in degree],
                            color='rgba(0,0,0,0.5)'#"crimson"
                        ),
                
                customdata= np.stack((G.vs['betweenness'], G.vs['degree']), axis=-1),
                hovertemplate ="<b>%{text}</b><br>"+\
                                "Betweenness: %{customdata[0]}<br>"+\
                                "Degree: %{customdata[1]}"
                                )
                                )
    fig.update_layout(title='', showlegend=False, autosize=False,hovermode='closest',
                width=1000, height=1000, margin=dict(l=10, r=0, t=20, b=0),
                paper_bgcolor = 'rgba(0,0,0,0)', plot_bgcolor = 'rgba(0,0,0,0)',
                annotations=[dict(showarrow=False, xref='paper', yref='paper', #text='Igraph representation in Plotly'
                x=0, y=-0.1, xanchor='left', yanchor='bottom', font=dict(size=14))], 
                xaxis=dict(showline=False, # hide axis line, grid, ticklabels and  title
                    zeroline=False, showgrid=False, showticklabels=False, title=''),
                yaxis=dict(showline=False, # hide axis line, grid, ticklabels and  title
                    zeroline=False, showgrid=False, showticklabels=False,title=''),
                )
    return (fig)


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
