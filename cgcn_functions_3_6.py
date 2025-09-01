import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
from py2neo import Graph
import authSettings as aS

# Initialize Neo4j connection
try:
    graph = Graph(aS.graphURL, auth=(aS.graphUser, aS.graphPass))
    print("Successfully connected to Neo4j database")
except Exception as e:
    print(f"Error connecting to Neo4j database: {e}")

from plotly.validators.layout import annotation
import sketch2Neo2 as s2n
import sentiment_functions_3_6 as sentiment_functions

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
import networkx as nx
import itertools




##################### Database queries
def struktura_korpusa():
    return s2n.corpusListDf

def databaza_izlistaj_strukturu_upita():
    return graph.run('match (n) return n limit 1 ').to_table()

def databaza_izlistaj_sve_pos_sketch(pos, language):
    '''stvara df sa svim imenicama iz x korpusa'''
    df= graph.run('''match (n:Lemma{language:$language})
            where n.lpos = '-'+$pos  
            return n.lempos as lempos 
            ''', pos=pos, language=language).to_data_frame()
    return df 

def databaza_izlistaj_sve_pos_sketch_corpus(corpusID, pos):
    '''stvara df sa svim imenicama iz x korpusa'''
    df= graph.run('''match (n:Lemma)
            where n[$freq]<>'' and n.lpos = '-'+$pos
            with n[$freq] as freq, n[$relFreq] as relFreq, n.lempos as lempos
            return lempos, freq, relFreq 
            order by freq desc''', pos=pos, freq= 'freq_'+corpusID, relFreq='relFreq_'+corpusID).to_data_frame()
    return df 

########################
# Frequency and Relative frequency of the source lemma in corpus
@st.cache_data
# def source_lemma_freq(lemma, pos, corpusID, corpus, language, gramRel):
#     'find frequency of the source lema, if not present harvest from the s2n function'
#     try:
#         q='''
#         match (n:Lemma{lempos:$lemma+$pos, language:$language})
#         where n[$freq]<>''
#         with n[$freq] as freq, n[$relFreq] as relFreq
#         return freq, relFreq
#         ''' 
#         df = graph.run(q, lemma= lemma, pos = pos, freq = 'freq_'+corpusID, relFreq='relFreq_'+corpusID, language= language).to_data_frame()        
#     except:
#         s2n.corpusSketchHarvest(cS, [lemma], aS.userName, aS.apiKey, "wsketch?", pos, pos, corpus, 1, language, gramRel)
#         df = graph.run(q, lemma= lemma, pos = pos, freq = 'freq_'+corpusID, language= language ).to_data_frame()
#     return (df)

def source_lemma_freq(lemma, pos, corpusID, corpus, language, gramRel):
    """
    Fetch the frequency of the source lemma. 
    If the lemma is not present in the database, harvest it using the s2n function.
    """
    # Define the query
    query = '''
    MATCH (n:Lemma{lempos: $lemma + $pos, language: $language})
    WHERE n[$freq] <> ''
    RETURN n[$freq] as freq, n[$relFreq] as relFreq
    '''
    params = {
        'lemma': lemma,
        'pos': pos,
        'freq': 'freq_' + corpusID,
        'relFreq': 'relFreq_' + corpusID,
        'language': language
    }
    
    try:
        df = graph.run(query, **params).to_data_frame()
        
        # If dataframe is empty, try to harvest the lemma
        if df.empty:
            s2n.corpusSketchHarvest(cS, [lemma], aS.userName, aS.apiKey, "wsketch?", pos, pos, corpus, 1, language, gramRel)
            df = graph.run(query, **params).to_data_frame()

    except Exception as e:
        print(f"Database error while trying to fetch lemma {lemma}: {e}")
        return None

    return df

# GramRel list - for a lemma in a corpus give a list of gramRels and count of collocation
@st.cache_data
#def lemmaGramRels(lemma, pos, corpusID, language):
#    try:    
#        q="""MATCH (n:Lemma{lempos:$lemma+$pos, language:$language})-[r]-(m:Lemma)
#        WHERE r[$count]<>""
#        WITH type(r) as rels
#        RETURN distinct(rels) as gramRels, count(rels) as count
#        """
#        df = graph.run(q, language=language, lemma=lemma, pos=pos, count='count_'+corpusID, score='score_'+corpusID).to_data_frame()
#    except:
#        pass
#    return(df)  //previous_values

@st.cache_data
def lemmaGramRels(lemma, pos, corpusID, language):
    """
    Get a list of grammatical relations and their counts for a given lemma in a corpus.
    Returns a DataFrame with 'gramRels' and 'count' columns.
    """
    try:    
        # First check if the lemma exists
        check_query = """
        MATCH (n:Lemma{lempos:$lemma+$pos, language:$language})
        RETURN count(n) as count
        """
        result = graph.run(check_query, lemma=lemma, pos=pos, language=language).to_data_frame()
        
        if result.iloc[0]['count'] == 0:
            # Lemma doesn't exist
            return pd.DataFrame({'gramRels': ['koordinacija'], 'count': [0]})
            
        # Only add "22" if it's hrwac and doesn't already have 22
        count_property = 'count_' + corpusID
        score_property = 'score_' + corpusID
        
        q = """
        MATCH (n:Lemma{lempos:$lemma+$pos, language:$language})-[r]-(m:Lemma)
        WHERE r[$count] <> ""
        WITH type(r) as rels
        RETURN distinct(rels) as gramRels, count(rels) as count
        ORDER BY count DESC
        """
        df = graph.run(
            q, 
            language=language, 
            lemma=lemma, 
            pos=pos, 
            count=count_property,
            score=score_property
        ).to_data_frame()
        
        # If DataFrame is empty, return a default one with at least one relation
        if df.empty:
            df = pd.DataFrame({'gramRels': ['koordinacija'], 'count': [0]})
            
    except Exception as e:
        print(f"Error in lemmaGramRels: {e}")
        # Return a default DataFrame with at least one relation
        df = pd.DataFrame({'gramRels': ['koordinacija'], 'count': [0]})
    
    return df


# GramReldf (function)
@st.cache_data
def lemmaByGramRel(language, lemma, pos, gramRel, corpusID, measure):
    #for a lemma, gramRel in a corpus get list of collocations
    try:
        # Use the corpus ID directly for property names
        count_property = 'count_' + corpusID
        score_property = 'score_' + corpusID
        
        q="""
        MATCH (n:Lemma{lempos:$lemma+$pos,language:$language})-[r]-(m:Lemma{language:$language})
        WHERE type(r) = $gramRel
        WITH n.lempos as source, r[$count] as freq, r[$score] as score, m.lempos as friend
        RETURN source, friend, freq, score, $gramRel
        """
        d = graph.run(q, 
                     language=language, 
                     lemma=lemma, 
                     pos=pos, 
                     gramRel=gramRel, 
                     count=count_property, 
                     score=score_property).to_data_frame()
        d = d.drop_duplicates().dropna().sort_values(by=[measure], ascending=False)
    except Exception as e:
        print(f"Error in lemmaByGramRel: {e}")
        return pd.DataFrame(columns=['source', 'friend', 'freq', 'score', gramRel])
    return d



# FriendCoocurences dataframe
st.cache_data
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


@st.cache_data
def is_network(lemposSeries, language, corpus, corpusID, is_gramRel, pos2, limit, measure, harvestSelect, direction):
    df=pd.DataFrame()
    for row in lemposSeries:
        try:
            is_friend= friends_df(language, row.split('-')[0], '-'+row.split('-')[1], corpus, corpusID, '"%w" is a ...', '-n', limit, measure, harvestSelect, 'directed2')
            df= pd.concat([df,is_friend])
        except:
            pass        
    return df


# Lemma network profile - for a lemma in a corpus gives a collocation profile with all gramRels
@st.cache_data
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
                        font_size=12, font_family='Verdana',
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
@st.cache_data
def FoFData(language, friendCoocurences, lemma, pos, corpus, corpusID, gramRel, pos2, limit, measure, harvestSelect, direction):
    #start with Friend dataset
    df_fof = friendCoocurences[['source', 'friend', 'freq','score','gramRel','corpus']]
    #extract friends of FoF network
    try:            
        for index, row in df_fof.iterrows():
            if direction=='undirected':
                df2= friends_df(language, row['friend'][0:-2], row['friend'][-2:], corpus, corpusID, gramRel, pos2, limit, measure, harvestSelect, direction)
            if direction=='directed':
                df2= friends_df(language, row['friend'][0:-2], row['friend'][-2:], corpus, corpusID, gramRel, pos2, limit, measure, harvestSelect, direction) #'directed2'        
            #append the dataframe of df2 to df
            df2= df2.rename(columns={'friend': 'source', 'source':'friend'})
            df_fof=pd.concat([df_fof, df2], sort=False)
               
    except:
        pass
    try:
        df0= friends_df (language, lemma, pos, corpus, corpusID, gramRel, pos2, limit, measure, harvestSelect, direction)
        df_fof=pd.concat([df_fof,df0], sort=False)
        
    except:
        pass
    df_fof=df_fof.drop_duplicates().dropna()
    return(df_fof)


# Friend IGraph calculations
@st.cache_data
def Fgraph(dataset, measure, direction):
    if direction == 'directed':
        d = True
        m = 'IN'
    else:
        d= False
        m = 'ALL'
    'calculate a graph from dataset of source - friend coocurences with a chosen measure'
    #create df variable
    df=dataset[['source', 'friend', measure]]
    #create tuples from df.values
    tuples = [tuple(x) for x in df.values]
    #create igraph object from tuples
    G=ig.Graph.TupleList(tuples, directed = d, edge_attrs=['weight'], vertex_name_attr='name', weights=False)
    #create vertex labels from name attr
    G.vs["label"]= G.vs["name"]
    G.vs["degree"]=G.vs.degree(mode = m)
    G.vs["pagerank"]=G.vs.pagerank(directed=d, weights='weight')
    G.vs["personalized_pagerank"]=G.vs.personalized_pagerank(directed=d, weights='weight')
    # weighted degree je kod neusmjerenog ALL, a kod usmjerenog IN
    G.vs["weighted_degree"] = G.strength(G.vs["label"], weights='weight', mode= m)
    G.vs["betweenness"] = G.betweenness()
    G.vs["color"] = "rgba(255,0,0,0.2)"

    # G.vs['eigen_c'] = G.eigenvector_centrality(directed=False, scale=True, weights= 'weight', return_eigenvalue=False)
    # G.vs["sbb_importance"] = sbb_importance(G, True)
    # G.vs["sbb2_importance"] = [(G.vs[j]["sbb_importance"]*G.vs[j]["weighted_degree"]) for j in range(0,len(G.vs))]
    return(G)


# Friend of Friend Igraph calculations
# @st.cache_data
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
    G.vs['betweenness'] = G.betweenness(vertices=None, directed=False, cutoff=None, weights=None)
    G.vs["pagerank"]=G.vs.pagerank(directed=False, weights='weight')
    G.vs["weighted_degree"] = G.strength(G.vs["label"], weights='weight', mode='ALL') #OUT|IN give the same measure on the nondirected 
    G.vs['sli']= sli_importance(G, igraph=True, normalize=False)
    G.vs['sli_norm']= normalize(G.vs['sli'])
    # G.vs["personalized_pagerank"]=G.vs.personalized_pagerank(directed=False, damping=0.85, reset=None, reset_vertices=None, weights='weight')
    #add weighted degree to vs: https://igraph.org/python/doc/igraph.GraphBase-class.html#strength
    G.vs['shape']="circle"
    G.vs["color"] = "rgba(255,0,0,0.2)"
    return(G)

@st.cache_data
def df_from_graph(G):
    'create a dataframe from graph and sort lempos'
    df= pd.DataFrame(columns=['label', 'degree', 'pagerank', 'weighted_degree', 'betweenness', 'sli'])# 'sbb_importance','sbb2_importance'
    try:
        df['label']=G.vs['name']
        df['degree']=G.vs['degree']
        df['pagerank']=G.vs['pagerank']
        df['weighted_degree']=G.vs['weighted_degree']
        df['betweenness']=G.vs['betweenness']
        df['sli']= G.vs['sli']
        df['sli_norm']=G.vs['sli_norm']
    except:
        pass
    return (df)


# Clustering algorithm
@st.cache_resource
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
#@st.cache_data
def AlgoDataFrame(FoFgraph, clusterAlgo):
    #create a FoF graph G
    G = FoFgraph
    G.vs['betweenness'] = G.betweenness(vertices=None, directed=True, cutoff=None, weights=None)
    G.vs["weighted_degree"] = G.strength(G.vs["label"], weights='weight', mode='ALL')
    #store graph measures
    partitions = clusterAlgo
    commsDataFrame=pd.DataFrame()
    #set community partition counter
    commPart=-1
    # loop through partitions and set properties
    for partition in partitions:
        try:
            commPart=commPart+1
            G.vs['commPart']=commPart
            commsDataFramePart=pd.DataFrame({'label': G.vs[partition]['label'],
                                             'commPart': G.vs[partition]['commPart'],
                                             'degree': G.vs[partition].degree(),
                                             'weighted_degree': G.vs[partition]['weighted_degree'],
                                             'pagerank': G.vs[partition].pagerank(),
                                             'sli': G.vs[partition]['sli'],
                                             'sli_norm': G.vs[partition]['sli_norm'],
                                             'betweenness': G.vs[partition]['betweenness']                                   
                                            #  'sbb_importance': G.vs[partition]['sbb_importance'],
                                            #  'sbb2_importance': G.vs[partition]['sbb2_importance'],
                                             })
            #for each partition and node in partition get some properties
            commsDataFramePart['pagerankPart_mean']=sum(G.vs[partition].pagerank())/len(G.vs[partition])
            commsDataFramePart['degreePart_mean']=sum(G.vs[partition].degree())/len(G.vs[partition])
            #append partition to dataframe
            commsDataFrame= pd.concat([commsDataFrame,commsDataFramePart])
        except:
            pass
    # add pagerank mean for all nodes
    commsDataFrame['pagerankAll_mean'] = commsDataFrame['pagerank'].mean()
    # add degree mean for all nodes
    commsDataFrame['degreeAll_mean']= commsDataFrame['degree'].mean()   
    return (commsDataFrame)




################################ Network plot 
# # https://towardsdatascience.com/tutorial-network-visualization-basics-with-networkx-and-plotly-and-a-little-nlp-57c9bbb55bb9

def make_edge(x, y, text, width, color):
    return  go.Scatter(x=x, y=y,
                       line = dict(width = abs(width),
                                   color = color, shape='spline', smoothing=1.3),
                       hovertext = 'text',
                       text      = ([text]),
                       mode      = 'lines',
                       )
def make_partition(x, y, text, width, color):
    return  go.Scatter(x=x, y=y,
                    howerinfo= 'x+y',
                       line = dict(width = width,
                                   color = color, shape='spline', smoothing=1.3),
                       hovertext = 'text',
                       text      = ([text]),
                       mode      = 'lines',
                       )
def make_partition_area(x, y):
    return  go.Scatter(x=x, y=y,
                mode='lines', 
                name='Area',
                fill='toself',
                line_color='rgba(0,0,0,0.1)',
                fillcolor= 'rgba(0,0,0,0.5)'
                            )

@st.cache_data
def FgraphDraW2_plotly(Fgraph, layout, vertexSize, vertexLabelSize, imageName, edgeLabelSize):   
    # vertex and edges atributes of Fgraph are treated as lists 
    G=Fgraph
    labels=G.vs['label']
    betweenness=G.vs['betweenness']
    degree = G.vs['degree']    
    # edgeSize = 2 # edge size for all
    N=len(G.vs) # number of nodes
    E=[e.tuple + (e['weight'],) for e in G.es]  # list of tuples containing (Source_node number, Target_node number) + added a edge weight in tuple
    # https://stackoverflow.com/questions/4913397/how-to-add-value-to-a-tuple
    layt=G.layout(layout) # layout https://igraph.org/python/doc/tutorial/tutorial.html#layouts-and-plotting
    # Create figure
    fig = go.Figure()

    ###################edges
    ## edges from the layout
    edge_trace=[] # collect traces
    Xe=[]
    Ye=[]
    c=112 # color
    for e in E: #for edge tuple in list of edges tuple
        c=c+1 # color
        text_on_edge= str(e[2])
        Xe=[layt[e[0]][0],layt[e[1]][0], None]
        Ye=[layt[e[0]][1],layt[e[1]][1], None]
        color='rgba(16,'+str(c)+',2,0.5)' # color
        trace  = make_edge(Xe, Ye, text_on_edge, 0.3*e[2]/math.log2(e[2]), color)
        edge_trace.append(trace)
    # Plot all edge traces
    for trace in edge_trace:
        fig.add_trace(trace)
    
    
    #################################nodes
    ## node x, y components of the network
    Xn=[layt[k][0] for k in range(N)]
    Yn=[layt[k][1] for k in range(N)]    
    fig.add_trace(go.Scatter(x=Xn, y=Yn,
                mode='markers+text', 
                name='Lemma',
                marker=dict(symbol='circle-dot', 
                            size= [math.log(1/(1/(i+4)))*vertexSize for i in betweenness],
                            color=G.vs['color'],
                            line=dict(color='rgba(16, 112, 2, 0.7)', 
                            width=0.5)
                                ),
                text=labels, 
                textfont=dict(
                            family="Verdana",
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
    fig.update_layout(title=imageName, showlegend=False, autosize=False, hovermode='closest', dragmode='pan',
                font_size=10, font_family='Verdana', 
                height=350,
                margin=dict(l=10, r=10, t=20, b=20),
                paper_bgcolor = 'rgba(0,0,0,0)', plot_bgcolor = 'rgba(255,255,250,0.5)',
                annotations=[dict(showarrow=False, xref='paper', yref='paper', 
                text='', x=0, y=-0.1, xanchor='left', yanchor='bottom', font=dict(size=14))], 
                xaxis=dict(showline=False, zeroline=False, showgrid=False, showticklabels=False, title=''),
                yaxis=dict(showline=False, zeroline=False, showgrid=False, showticklabels=False, title=''),
                )   
    return (fig)



from plotly_resampler import FigureResampler, FigureWidgetResampler



@st.cache_data
def FgraphDraW_plotly(Fgraph, layout, vertexSize, vertexLabelSize, imageName, edgeLabelSize):   
    # vertex and edges atributes of Fgraph are treated as lists 
    G=Fgraph
    labels=G.vs['label']
    betweenness=G.vs['betweenness']
    degree = G.vs['degree']
    edgeSize = 2 # edge size for all
    N=len(labels) # number of nodes
    E=[e.tuple for e in G.es]# list of edges containing 
    layt=G.layout(layout) # layout https://igraph.org/python/doc/tutorial/tutorial.html#layouts-and-plotting
    ## node x, y components of the network
    Xn=[layt[k][0] for k in range(N)]
    Yn=[layt[k][1] for k in range(N)]
    ## edges from the layout
    Xe=[]
    Ye=[]
    for e in E:
        Xe+=[layt[e[0]][0],layt[e[1]][0], None]
        Ye+=[layt[e[0]][1],layt[e[1]][1], None]    
    
    
    fig=go.Figure(go.Scatter(x=Xe, y=Ye,
                mode='lines', name='Edges',
                line= dict(width=edgeSize, color='rgba(16, 112, 2, 0.5)', shape='spline', smoothing=1.3),
                hoverinfo='none'
                ))

    for e in E:
        fig.add_trace(go.Scatter(x=[layt[e[0]][0], layt[e[1]][0], None],
                                y=[layt[e[0]][1], layt[e[1]][1], None],
                                mode='lines',
                                name='Edges',
                                line=dict(width=1)))
    
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
                            family="Verdana",
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
    fig.update_layout(go.Layout(title=imageName, 
                showlegend=False, autosize=True, hovermode='closest', dragmode= 'pan',
                font_size=10, font_family='IBM Plex Sans', 
                height=350,
                margin=dict(l=10, r=10, t=20, b=20),
                paper_bgcolor = 'rgba(0,0,0,0)', plot_bgcolor = 'rgba(255,255,250,0.5)',
                annotations=[dict(showarrow=False, xref='paper', yref='paper', 
                text='', x=0, y=-0.1, xanchor='left', yanchor='bottom', font=dict(size=14))], 
                xaxis=dict(showline=False, zeroline=False, showgrid=False, showticklabels=False, title=''),
                yaxis=dict(showline=False, zeroline=False, showgrid=False, showticklabels=False, title=''),
                ))   
    return (fig)


####################### Clustered Second degree network graph
# @st.cache_data
def FgraphDraW_plotly_cluster(graph, partition, layout, vertexSize, vertexLabelSize, imageName, edgeLabelSize):
    # vertex and edges atributes of Fgraph are treated as lists 
    G=graph
    G.vs['membership']= partition.membership #[x+1 for x in partition.membership] # dodali 1 da se vidi bez nula
    labels=G.vs['label']
    betweenness=G.vs['betweenness']
    degree = G.vs['degree']
    layt=G.layout(layout)
    
    #Cluster Colors Programatic Method
    # palette = ig.drawing.colors.ClusterColoringPalette(len(partition))
    palette = ig.drawing.colors.GradientPalette("red", "blue", len(partition))
    pal_colors = palette.get_many(partition.membership)
    transparency = 0.2 # color transparency
    # Replace a transparency value in tuple ex. (1,1,0,1) > (1,1,0,0.5)
    graph_colors=[] 
    for col in palette:
        cl=list(col) #convert tuples to list
        cl[3]= transparency #add opacity value
        graph_colors.append(tuple(cl)) 

    for part in range(len(partition)):
        part_vs=G.vs.select(membership_eq = str(part))
        part_es=G.es.select(_within = part_vs)
        part_es['color']= ['rgba'+str(x) for x in graph_colors]
    
    ###################trace edges
    edge_trace=[] # collect traces in a list of tuples
    E=[e.tuple + (e['weight'],)+ (e['color'],) for e in G.es]# list of edges
    Xe=[]
    Ye=[]
    for e in E: #for edge tuple in list of edges tuple
        text_on_edge= str(e[2])
        Xe=[layt[e[0]][0],layt[e[1]][0], None]
        Ye=[layt[e[0]][1],layt[e[1]][1], None]
        color= 'rgba'+str(graph_colors[0])# 'rgba('+str(e[3])+')'#'rgba(0,'+str(c)+',0,0.5)' # color
        trace  = make_edge(Xe, Ye, text_on_edge, 0.4*e[2], color)
        edge_trace.append(trace)
    
    # Create figure
    fig = go.Figure()
    # Plot all edge traces
    for trace in edge_trace:
        fig.add_trace(trace)


    ## node x, y components of the network
    N=len(labels) # number of nodes
    Xn=[layt[k][0] for k in range(N)]
    Yn=[layt[k][1] for k in range(N)]
     
    fig.add_trace(go.Scatter(x=Xn, y=Yn,
                mode='markers+text', 
                name='Lemma',
                marker=dict(symbol='circle-dot', 
                            # size= [math.log(1/(1/(i+4)))*vertexSize for i in betweenness],
                            size= [i*vertexSize for i in G.vs['sli_norm']],
                            color=G.vs['membership'],
                            line=dict(color='white', width=1)),
                text=labels, 
                textfont=dict(family="Verdana",
                            size=[math.log(1/(1/(i+5)))*vertexLabelSize*0.6 for i in degree],
                            color='rgba(0,0,0,0.9)'
                            
                            ),
                customdata= np.stack((  G.vs['betweenness'], 
                                        G.vs['degree'],
                                        G.vs['membership']), axis=-1),
                hovertemplate ="<b>%{text}</b><br>"+\
                                "Betweenness: %{customdata[0]}<br>"+\
                                "Degree: %{customdata[1]}<br>"+\
                                "Partition:%{customdata[2]}"
                            )
                    ) 


    fig.update_layout(title='', showlegend=False, autosize=True,hovermode='closest', dragmode='pan',
                font_size=10, font_family='Verdana',
                width=500, height=550, 
                margin=dict(l=10, r=10, t=10, b=0),
                paper_bgcolor = 'rgba(0,0,0,0)', plot_bgcolor = 'rgba(255,255,250,0.5)',
                annotations=[dict(showarrow=False, xref='paper', yref='paper', 
                text='Network', x=0, y=-0.1, xanchor='left', yanchor='bottom', font=dict(size=14))], 
                xaxis=dict(showline=False, zeroline=False, showgrid=False, showticklabels=False, title=''),
                yaxis=dict(showline=False, zeroline=False, showgrid=False, showticklabels=False,title=''),
                )
    return (fig)


####################### Clustered 3D Second degree network graph
# @st.cache_data
def FgraphDraW_plotly_cluster3d(graph, partition, layout, vertexSize, vertexLabelSize, imageName, edgeLabelSize):
    # vertex and edges atributes of Fgraph are treated as lists 
    G=graph
    G.vs['membership']= partition.membership #[x+1 for x in partition.membership] # dodali 1 da se vidi bez nula
    labels=G.vs['label']
    betweenness=G.vs['betweenness']
    degree = G.vs['degree']
    layt=G.layout("fruchterman_reingold", dim=3) # https://igraph.org/python/doc/tutorial/tutorial.html#layouts-and-plotting
    
    #Cluster Colors Programatic Method
    # palette = ig.drawing.colors.ClusterColoringPalette(len(partition))
    palette = ig.drawing.colors.GradientPalette("red", "blue", len(partition))
    pal_colors = palette.get_many(partition.membership)
    transparency = 0.2 # color transparency
    # Replace a transparency value in tuple ex. (1,1,0,1) > (1,1,0,0.5)
    graph_colors=[] 
    for col in palette:
        cl=list(col) #convert tuples to list
        cl[3]= transparency #add opacity value
        graph_colors.append(tuple(cl)) 

    for part in range(len(partition)):
        part_vs=G.vs.select(membership_eq = str(part))
        part_es=G.es.select(_within = part_vs)
        part_es['color']= ['rgba'+str(x) for x in graph_colors]
    
    ###################trace edges
    edge_trace=[] # collect traces in a list of tuples
    E=[e.tuple + (e['weight'],)+ (e['color'],) for e in G.es]# list of edges
    Xe=[]
    Ye=[]
    Ze=[]
    for e in E: #for edge tuple in list of edges tuple
        text_on_edge= str(e[2])
        Xe=[layt[e[0]][0],layt[e[1]][0], None]
        Ye=[layt[e[0]][1],layt[e[1]][1], None]
        Ze=[layt[e[0]][2],layt[e[1]][1], None]
        color= 'rgba'+str(graph_colors[0])# 'rgba('+str(e[3])+')'#'rgba(0,'+str(c)+',0,0.5)' # color
        # trace  = make_edge(Xe, Ye, text_on_edge, 0.4*e[2], color)
        # edge_trace.append(trace)
    
    # edges
    fig=go.Figure(go.Scatter3d(x=Xe, y=Ye,z=Ze,
                mode='lines',
                line= dict(width=1, color='rgba(255, 112, 2, 0.5)', 
                # shape='spline', smoothing=1.3
                ),
                hoverinfo='none'))

    # Create figure
    fig = go.Figure()
    # Plot all edge traces
    for trace in edge_trace:
        fig.add_trace(trace)


    ## node x, y components of the network
    N=len(labels) # number of nodes
    Xn=[layt[k][0] for k in range(N)]
    Yn=[layt[k][1] for k in range(N)]
    Zn=[layt[k][2] for k in range(N)]
     
    fig.add_trace(go.Scatter3d(x=Xn, y=Yn, z=Zn,
                mode='markers+text', 
                name='Lemma',
                marker=dict(symbol='circle', 
                            # size= [math.log(1/(1/(i+4)))*vertexSize for i in betweenness],
                            size= [i*vertexSize for i in G.vs['sli_norm']],
                            color=G.vs['membership'],
                            line=dict(color='white', width=1)),
                text=labels, 
                textfont=dict(family="Verdana",
                            size=[math.log(1/(1/(i+5)))*vertexLabelSize*0.6 for i in degree],
                            color='rgba(0,0,0,0.9)'
                            
                            ),
                customdata= np.stack((  G.vs['betweenness'], 
                                        G.vs['degree'],
                                        G.vs['membership']), axis=-1),
                hovertemplate ="<b>%{text}</b><br>"+\
                                "Betweenness: %{customdata[0]}<br>"+\
                                "Degree: %{customdata[1]}<br>"+\
                                "Partition:%{customdata[2]}"
                            )
                    ) 


    fig.update_layout(title='', showlegend=False, autosize=True, hovermode='closest', dragmode='orbit',
                font_size=10, font_family='Verdana',
                width=500, height=550, 
                margin=dict(l=10, r=10, t=10, b=0),
                paper_bgcolor = 'rgba(0,0,0,0)', plot_bgcolor = 'rgba(255,255,250,0.5)',
                annotations=[dict(showarrow=False, xref='paper', yref='paper', 
                text='Network', x=0, y=-0.1, xanchor='left', yanchor='bottom', font=dict(size=14))], 
                xaxis=dict(showline=False, zeroline=False, showgrid=False, showticklabels=False, title=''),
                yaxis=dict(showline=False, zeroline=False, showgrid=False, showticklabels=False,title=''),
                )
    return (fig)


# Create json structure for the 3d representation
def json_FoF_clustered(df_fof, commsDataFrame, measure):
    df_nodes = commsDataFrame[['label', 'commPart']].rename(columns={'label':'id', 'commPart':'group'}).to_json(orient="records")
    df_links = pd.DataFrame(df_fof[['source','friend', measure]]).reset_index().drop(columns=['index']).rename(columns={'source':'source', 'friend':'target', measure:'value'}).to_json(orient="records")
    df='{"nodes":'+df_nodes+',"links":'+df_links+'}'
    return df


st.cache_data
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








def sli_importance(G, **kwargs):
    '''
    The algorithm takes graph G and get the importance of the nodes as a list of floating values
    kwargs: 
        igraph (default False), if True transforms G from igraph to Networkx
        normalize (default True), if False returns non-normalized values from a list of SLI importance values
    '''
    # G introduced in igraph module is transformed into networkx module. If you want to introduce this feature you have to import igraph module
    if kwargs.get('igraph')==True:
        # import igraph
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

        counter_connect=0
        
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
            nodeWeight_node = g.nodes[node]["weighted_degree"]
            nodeWeight_neigh =  g.nodes[neigh]["weighted_degree"]
            p = dict1[(node, neigh)] # broj ciklusa u kojem sudjeluje edge (node, susjed)
            u = (nodeWeight_node+ nodeWeight_neigh - 2* edge_weight)
            lambd = p+1
            z = nodeWeight_node/ (nodeWeight_node+ nodeWeight_neigh)*edge_weight
            I = u*lambd*z
            sum = sum + I
        SLI_importance.append(sum + nodeWeight_node)
    SLI_importance= pd.Series(SLI_importance)
    
    
    # SLI values non-normalized
    if kwargs.get('normalize') == False:
        SLI_importance_result = SLI_importance
    # SLI values normalized as default
    else:
        SLI_importance_normalized = SLI_importance/SLI_importance.sum()*100
        SLI_importance_result = SLI_importance_normalized 
    return SLI_importance_result


def normalize(list_values):
    #uzima Series i vraća series normaliziran
    list_values= pd.Series(list_values)
    sum= list_values.sum()
    normalized_list_values= list_values/sum*100
    return normalized_list_values


def srednjica(val_list, node_importance_values_list):
    # takes the list of valency values and node importance values
    try:
        brojnik=[]
        nazivnik= [ float(x) for x in node_importance_values_list]
        
        for item1 in zip(val_list,node_importance_values_list):
            brojnik.append(float(item1[0])*float(item1[1]))
        
        result = sum(brojnik)/sum(nazivnik)
        return result
        
    except Exception as e:
        return 0.0


@st.cache_data
def df_from_graph(G):
    'create a dataframe from graph and sort lempos'
    df= pd.DataFrame(columns=['label', 'degree', 'pagerank', 'weighted_degree', 'betweenness', 'eigen_c'])
    try:
        df['label']=G.vs['name']
        df['degree']=G.vs['degree']
        df['pagerank']=G.vs['pagerank']
        df['weighted_degree']=G.vs['weighted_degree']
        df['betweenness']=G.vs['betweenness']
        df['sli']=G.vs['sli']
        # df['eigen_c']=G.vs['eigen_c']
    except:
        pass
    return (df)


@st.cache_data
def lemmaGramRelLemmas(lemma, pos, gramRel, corpusID, language):
    """
    Get a list of lemmas connected by a specific grammatical relation to a given lemma in a corpus.
    Returns a DataFrame with 'lempos', 'count', and 'score' columns.
    """
    try:
        # Use the corpus ID directly for property names
        count_property = 'count_' + corpusID
        score_property = 'score_' + corpusID
        
        q = """
        MATCH (n:Lemma{lempos:$lemma+$pos, language:$language})-[r:$gramRel]-(m:Lemma)
        WHERE r[$count] <> ""
        RETURN m.lempos as lempos, r[$count] as count, r[$score] as score
        ORDER BY count DESC
        """
        df = graph.run(
            q, 
            language=language, 
            lemma=lemma, 
            pos=pos, 
            gramRel=gramRel,
            count=count_property,
            score=score_property
        ).to_data_frame()
        
        # If DataFrame is empty, return a default one
        if df.empty:
            df = pd.DataFrame({'lempos': [], 'count': [], 'score': []})
            
    except Exception as e:
        print(f"Error in lemmaGramRelLemmas: {e}")
        # Return an empty DataFrame
        df = pd.DataFrame({'lempos': [], 'count': [], 'score': []})
    
    return df
