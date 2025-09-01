# -*- coding: utf-8 -*-


'''
Ovo je verzija cgcn streamlit appa u kojoj je:
1) stavljeno Sentiment calculations

'''
# Imports
import authSettings as aS
import sketch2Neo2 as s2n
from py2neo import Graph
import cgcn_functions_3_5_sbb as cgcn_functions
import wordnet_functions
import spacy_wordnet_functions as spwf
import math
import time
import pandas as pd
import json
# desired_width=500
# pd.set_option('display.width', desired_width)
# pd.set_option('display.max_columns', 100, 'display.max_rows', 100)

import numpy as np
from plotly.offline import plot
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px
import time
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

# from streamlit_echarts import st_echarts
# from streamlit_echarts import st_pyecharts
# from pyecharts import options as opts
# from pyecharts.charts import Bar
# from pyecharts.charts import Bar3D
# from pyecharts.charts import Graph as GraphPy #there can be only 1 Graph
from PIL import Image

#https://igraph.org/python/doc/python-igraph.pdf
import igraph as ig
import louvain
import leidenalg
import networkx as nx 

# Simple persistent state: The dictionary returned by `get_state()` will be
# persistent across browser sessions.
@st.cache(allow_output_mutation=True)
def get_state():
    return {}

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

# Markdown
def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()


st.set_page_config(layout="wide")

# Title
st.markdown(read_markdown_file("markdown/introduction.md"), unsafe_allow_html=True)

# Corpus selection
# st.markdown(read_markdown_file("markdown/corpus_selection.md"), unsafe_allow_html=True)

with st.beta_container():
    # Corpus select
    cS= st.selectbox('Corpus', s2n.corpusListDf['cS'], key='corpus')
    # Corpus data based on the selection
    try:
        corpSelect=s2n.corpusSelect(cS)
        corpus=corpSelect[1]
        language=corpSelect[2]
        corpusID=corpSelect[3]
        gramRel=corpSelect[4] #initial gramRel coordination type
        initial_lexeme= corpSelect[5] # initial lexeme
        initial_pos= corpSelect[6]
    except:
        pass
    # st.markdown(read_markdown_file("markdown/lemma_selection.md"), unsafe_allow_html=True)
    lemma_column, pos_column = st.beta_columns((3,1))
    with lemma_column:
        # Lemma pos select
        try:
            lemma = st.text_input('Lemma', initial_lexeme,  key='lemma')
        except:
            pass
    with pos_column:
        # Lemma pos select
        try:
            pos = '-'+st.text_input('Part of speech',  initial_pos, key='pos')
        except:
            pass
    lempos = lemma+pos

####### Todo WIP - corpus info from api
# st.title('Corpus info')
# '', s2n.corpus_info(corpSelect[3], aS.userName, aS.apiKey)

##########################################################


# Source lexeme
# st.markdown("<h2 style='background-color:rgb(255, 153, 51);color:white;text-align:lefttext-align:left;padding: 10px 20px 10px 20px'><img src='http://emocnet.uniri.hr/wp-content/uploads/2020/09/017-distribution-300x300.png' alt='alt text' width=30>&nbsp;&nbsp;Source lexeme</h2>", unsafe_allow_html=True)
# st.title('Source lexeme')
# '##', lempos
# 'Corpus: ', corpSelect[0] 

assert lemma != '', st.error('Please choose a lemma')
assert pos != '-', st.error('Please choose a pos')


st.sidebar.markdown(read_markdown_file("markdown/network_parameters.md"), unsafe_allow_html=True)
# st.sidebar.subheader('Network parameters')
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
source_lema_data = cgcn_functions.source_lemma_freq(lemma, pos, corpusID, corpus, language, gramRel)


try:

    '**Frequency**: ', source_lema_data['freq'].values[0], '**Relative frequency (per million)**: ', source_lema_data['relFreq'].values[0]
except:
    pass


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




####### Wordnet integration
def print_val(word):
    return json.dumps(wordnet_functions.get_valency_for_lempos([word])).replace('"','').replace('{', '').replace('}', '')

st.markdown("<h2 style='text-align:left;font-size:25px; padding-left:20px'><img src='http://emocnet.uniri.hr/wp-content/uploads/2021/02/037-pyramid_chart-1.png' alt='alt text' width=30>&nbsp;&nbsp;<b>Wordnet senses</b></h2>", unsafe_allow_html=True)
with st.beta_expander('WordNet synsets'):
    # get synsets and hypernyms
    wn_hypernym=wordnet_functions.get_hypernyms([lempos], language)
    # get definitions
    wn_hypernym['s_definition']= [wordnet_functions.get_definition(str(x)[8:-2]) for x in wn_hypernym['source']]
    wn_hypernym['t_definition']= [wordnet_functions.get_definition(str(x)[8:-2]) for x in wn_hypernym['target']]
    
    #header
    cols_h = st.beta_columns((1,2,1,1,2,1))
    cols_h[0].write('**Sense**')
    cols_h[1].write('**Definition**')
    cols_h[2].write('**Valency**')
    cols_h[3].write('**Hypernym**')
    cols_h[4].write('**Definition**')
    cols_h[5].write('**Valency**')
    #content   
    for i in range(0, len(wn_hypernym)):
        cols = st.beta_columns((1,2,1,1,2,1))
        cols[0].markdown(f"<p style='font-size:1vw'><b>{str(wn_hypernym.source[i])[8:-2]}</b></p>", unsafe_allow_html=True)
        cols[1].markdown(f"<p style='font-size:1vw'><i>{wn_hypernym.s_definition[i]}</i></p>", unsafe_allow_html=True)
        cols[2].markdown(f"<p style='font-size:0.8vw'>{print_val(str(wn_hypernym.source[i])[8:-7])}</p>", unsafe_allow_html=True )
        cols[3].markdown(f"<p style='font-size:1vw'><b>{str(wn_hypernym.target[i])[8:-2]}</b></p>", unsafe_allow_html=True)
        cols[4].markdown(f"<p style='font-size:1vw'><i>{wn_hypernym.t_definition[i]}</i></p>", unsafe_allow_html=True)
        cols[5].markdown(f"<p style='font-size:0.8vw'>{print_val(str(wn_hypernym.target[i])[8:-7])}</p>", unsafe_allow_html=True)
            
with st.beta_expander('WordNet domains'):
    unique_syns_words=wn_hypernym['source'].astype(str).str.slice(start=8, stop=-7).unique()# unique words
    for syn_word in unique_syns_words:
        syn_domains= ', '.join(spwf.get_domains_for_word(syn_word))
        cols= st.beta_columns((1,7.3))
        cols[0].markdown(f"<p style='font-size:1vw'><b>{syn_word}</b>", unsafe_allow_html=True) 
        cols[1].markdown(f"<p style='font-size:1vw'>{syn_domains}", unsafe_allow_html=True)
    




########## GramRel list - for a lemma in a corpus give a list of gramRels and count of collocation
##################### Network construction (lemma, pos, gramRel, measure, pos2)
st.markdown("<h2 style='text-align:left;font-size:25px; padding-left:20px'><img src='http://emocnet.uniri.hr/wp-content/uploads/2020/09/020-algorithm-300x300.png' alt='alt text' width=30>&nbsp;&nbsp;<b>Network construction</b></h2>", unsafe_allow_html=True)

# store list of grammatical relations for a lemma
listGramRels= cgcn_functions.lemmaGramRels(lemma, pos, corpusID, language)   
# GramRel list option
with st.beta_expander(str(len(listGramRels))+' grammatical relations for source lexeme: '+lemma+pos):
    # view list of grammatical relations   
    fig = px.bar(listGramRels.sort_values(by='count', ascending=True), 
                x="count", y="gramRels", orientation='h', opacity=0.7, log_x=False, height=600, 
                color='count', color_continuous_scale="blugrn",
                text='count')
    fig.update_layout(
    paper_bgcolor = 'rgba(0,0,0,0)', plot_bgcolor = 'rgba(240,240,240,0.5)',
    font_family="IBM Plex Sans",
    font_color="black",
    font_size=10,
    title_font_family="IBM Plex Sans",
    title_font_color="red",
    legend_title_font_color="green",
    coloraxis_showscale=False)
    fig.update_xaxes(title_font_family="IBM Plex Sans")
    fig.update_layout(margin=dict(l=10, r=10, t=15, b=10))
    config={"displayModeBar": False, "showTips": False}
    st.plotly_chart(fig, use_container_width=True, config=config)




############### Network Visualization parameters
st.sidebar.markdown(read_markdown_file("markdown/visualization_parameters.md"), unsafe_allow_html=True)
# Vertex size selection 
vertexSizeType = st.sidebar.selectbox('Vertex size by', ('weighted_degree', 'degree', 'freq'))
if vertexSizeType == 'weighted_degree':
    if measure == 'score':
        vertexSizeValues= (0.0, 10.0, 5.0)
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
# Edge size selection
edgeSize = st.sidebar.slider('Edge size', 0.0,10.0,1.0)
# Edge label size selection
edgeLabelSize = st.sidebar.slider('Edge label size', 0.0,10.0,1.0)
# Layout selection
layout= st.sidebar.selectbox('Layout type', ('fr', 'kk', 'lgl', 'circle', 'drl', 'random', 'tree'))
# more layouts to come #####################

# Lemma network profile - for a lemma in a corpus gives a collocation profile with all gramRels
# with st.beta_container
# with st.beta_expander('Collocation list for '+lemma+pos):
#     lemmaProfile=st.checkbox('Collocation list for '+lemma+pos+':', key= 'lemmaProfile')
#     if lemmaProfile:
#         df= cgcn_functions.lemmaGramRelProfile(lemma, pos, corpusID, language)
#         '', df.dropna()


# Set minimal frequency and score parameters
minParameters=st.sidebar.checkbox('Set minimal frequency and score')
if minParameters:
    scoreMin=st.number_input('Minimum score', 0)
    freqMin=st.number_input('Minimum frequency', 0)

# Function for automatic direction selection
def direction(gramRel, initial):
    'define direction of the graph based on the gramRel'
    if gramRel == initial:
        direction = 'undirected'
    else:
        direction = 'directed'
    return (direction)



# GramRel selection (selectbox, selection based on the listGramRels df, initial is set from the corpSelect[4])
gramRel1_col, gramRel2_col = st.beta_columns(2)
with gramRel1_col:
    gramRel= st.selectbox('Grammatical relation 1: ('+lemma+pos+')-[gr1]-(f)', listGramRels['gramRels'], list(listGramRels['gramRels']).index(corpSelect[4]))
    # GramReldf (function)
    gramReldf= cgcn_functions.lemmaByGramRel(language, lemma, pos, gramRel, corpusID, measure)
    # Filter pos2 (selectbox from gramReldf) 
    pos_f= gramReldf['friend'].str[-2:].unique()
    # check if there are friends with multiple pos and choose 
    if len(pos_f)>1:
        pos2='-'+st.selectbox('Filter the pos value of the friend lexeme', gramReldf['friend'].str[-1].unique() )
    else:
        pos2=str(pos_f[0])
with gramRel2_col:
    gramRel2= st.selectbox('Grammatical relation 2: ('+lemma+pos+')-[gr1]-(f)-[gr2]-(fof)', listGramRels['gramRels'], list(listGramRels['gramRels']).index(corpSelect[4]))
    

############### First degree network
# activate function friends_df() and store as friendCoocurences variable
friendCoocurences = cgcn_functions.friends_df(language, lemma, pos, corpus, corpusID, gramRel, pos2, limit, measure, harvestSelect, direction(gramRel,corpSelect[4]))
with st.beta_expander('1st degree: ('+lemma+pos+')-['+gramRel+']-(f), '+str(len(friendCoocurences))+' nodes ranked by: '+measure+', direction: '+direction(gramRel,corpSelect[4])):
    # 2 columns: graph, scatter    
    f_vis_col1, f_vis_col2 = st.beta_columns((1,1.8))
    with f_vis_col1:
        Fgraph = cgcn_functions.Fgraph(friendCoocurences, measure) 
        # Draw Network representation
        config={"displayModeBar": False, "showTips": False}
        st.plotly_chart(cgcn_functions.FgraphDraW2_plotly(Fgraph, layout, vertexSize, vertexLabelSize, 'First degree network', edgeLabelSize),
                        use_container_width=True, 
                        config=config)
    with f_vis_col2:
        # Draw Friends Score vs Frequency Rank
        cgcn_functions.drawFriendCoocurences(friendCoocurences)        
    # Dataset selection
    if st.checkbox('1st degree dataset', value=False):
        '',friendCoocurences[['source','friend', 'freq', 'score', 'gramRel']]

#################################### 2nd degree Friend-of-friend network
# Store Friend-of-friend network dataframe in a df_fof variable
df_fof = cgcn_functions.FoFData(language, friendCoocurences, lemma, pos, corpus, corpusID, gramRel2, pos2, limit, measure, harvestSelect, direction(gramRel2, corpSelect[4]))
#store the result in a FoFgraph variable
FoFgraph=cgcn_functions.FoF_graph(df_fof, measure)    
with st.beta_expander('2nd degree: ('+lemma+pos+')-['+gramRel+']-(f)-['+gramRel2+']-(fof), '+ str(len(df_fof))+' nodes ranked by: '+measure):
    # visualize network
    # if st.checkbox('FoF graph visualization'):
    st.plotly_chart(cgcn_functions.FgraphDraW_plotly(
        FoFgraph, layout, vertexSize, vertexLabelSize, 'Second degree network', edgeLabelSize),
        use_container_width=True, config=config)

        # st.image(fof_image, caption='Friends-of-friends graph for '+lemma+pos
        #     +' with vertices: '+str(FoFgraph.vcount())
        #     +', edges: '+str(FoFgraph.ecount())
        #     +', max degree: '+str(FoFgraph.maxdegree(vertices=None, loops=False))
        #     +', graph density: '+str(FoFgraph.density(loops=False))
        #     +', average_path_length: '+str(FoFgraph.average_path_length())
        #     +', independence: '+str(FoFgraph.alpha())
        #     +', diameter: '+str(FoFgraph.diameter(directed=False, unconn=True, weights=None))
        #     , use_column_width=True)
    # dataset
    if st.checkbox('2nd degree dataset'):
        '', df_fof#[['source','friend', 'freq', 'score', 'gramRel']]



# Clustering 
st.markdown("<h2 style='text-align:left;font-size:25px; padding-left:20px'><img src='http://emocnet.uniri.hr/wp-content/uploads/2020/09/015-diagram-300x300.png' alt='alt text' width=30>&nbsp;&nbsp;<b>Clustering</b></h2>", unsafe_allow_html=True)
st.sidebar.markdown(read_markdown_file("markdown/clustering_parameters.md"), unsafe_allow_html=True)
# st.sidebar.subheader('Clustering parameters')
# algorithm selection 
algorithm = st.sidebar.selectbox('Cluster algorithm type', ('leiden', 'louvain'))
partitionType= st.sidebar.selectbox('Partition type', ('mvp', 'cpm'))
if partitionType == 'cpm':
    resolution = st.sidebar.slider('Resolution', 0.0, 1.0, 0.5)
else:
    resolution = None

# Identify clusters in FoFgraph
clusterAlgo = cgcn_functions.clusterAlgo(FoFgraph, algorithm, partitionType, resolution)

with st.beta_expander('Clustered 2nd degree network with '+str(len(FoFgraph.vs))+' nodes and '+str(len(clusterAlgo))+\
                    ' clusters using algorithm = '+algorithm+\
                    ', partition = '+partitionType+\
                    ', resolution = '+str(resolution)+'.'
                    ):

    # # cluster visualization
    config = dict({'scrollZoom': True, 'displaylogo': False, 'displayModeBar': False})
    st.plotly_chart(cgcn_functions.FgraphDraW_plotly_cluster(FoFgraph, clusterAlgo, layout, vertexSize, vertexLabelSize, 'clustered FoF graph', edgeLabelSize), use_container_width=True,config=config)
    #         +', edges: '+str(FoFgraph.ecount())
    #         +', max degree: '+str(FoFgraph.maxdegree(vertices=None, loops=False))
    #         +', graph density: '+str(FoFgraph.density(loops=False))
    #         +', average_path_length: '+str(FoFgraph.average_path_length())
    #         +', independence: '+str(FoFgraph.alpha())
    #         +', diameter: '+str(FoFgraph.diameter(directed=False, unconn=True, weights=None))
    #         , use_column_width=True)
    
    
    # if st.checkbox('Clustered Second degree (FoF) network list'):
    #     '', clusterAlgo



    # Explore communities
    st.subheader('Community exploration')
    
    FoFgraph.vs['sbb_importance']= cgcn_functions.sbb_importance(FoFgraph)
    FoFgraph.vs['sbb2_importance']= cgcn_functions.sbb2_importance(FoFgraph)
    FoFgraph.vs['sbb3_importance']= cgcn_functions.sbb3_importance(FoFgraph)
    FoFgraph.vs['sbb3_norm']= cgcn_functions.normalize(FoFgraph.vs['sbb3_importance'])
    FoFgraph.vs['shishi_importance']= cgcn_functions.shishi_importance(FoFgraph)
    FoFgraph.vs["pagerank"]=FoFgraph.vs.pagerank(directed=False, weights='weight')
    FoFgraph_df = pd.DataFrame({attr: FoFgraph.vs[attr] for attr in FoFgraph.vertex_attributes()})
    'FoFgraph_df', FoFgraph_df
    
    
    commsDataFrame=cgcn_functions.AlgoDataFrame(FoFgraph, clusterAlgo)
    algo_sort_by = st.multiselect('Sort nodes by', commsDataFrame.columns)
    try:    
        st.write(commsDataFrame.sort_values(by=algo_sort_by, ascending=False))
    except:
        pass
    




# ##### 3 d visualisation using the # https://github.com/vasturiano/3d-force-graph 
# # create json data of nodes, groups and links
# json_data =cgcn_functions.json_FoF_clustered(df_fof, commsDataFrame, measure)
# # 'Proba json', json_data 
# # Write json data in the datasets folder - this method is not ok for production due to the writing - the data should be stored in memory
# out_file=open('3d-force-graph/example/datasets/border.json', 'w')
# out_file.write(json_data)
# out_file.close()

# # 3d Graph as a iframe
# # stranica se mora vrtiti : ako je na virtualnoj desni klik  Open with live server
# components.iframe('http://127.0.0.1:5501/3d-force-graph/example/tree/index.html', height=500)



# #################### cluster 2nd degree friends by coordination
# # identify source-friends network using a grammatical relation (not coordination) and cluster friends using f_graph with relation=coordination
# def f_connected_by_coordination(lemma, pos, corpus, gramRel, initial, pos2, limit, coord_numb, measure, harvestSelect):
#     # get friends df
#     df = cgcn_functions.friends_df(language, lemma, pos, corpus, corpusID, gramRel, pos2, limit, measure, harvestSelect, direction(gramRel, corpSelect[4]))[['source', 'friend', 'freq','score','gramRel','corpus']]
#     # get friend_of_friends_by_coordination
#     try:            
#         for index, row in df.iterrows():
#             df2= cgcn_functions.friends_df(language, row['friend'][0:-2], row['friend'][-2:], corpus, corpusID, initial, pos2, coord_numb, measure, harvestSelect, direction(gramRel, initial))        
#             #append the dataframe of df2 to df
#             df2= df2.rename(columns={'friend': 'source', 'source':'friend'})
#             df=df.append(df2, sort=False)
#             df=df.drop_duplicates().dropna()   
#     except:
#         pass
    
#     # create a FoF graph
#     G= cgcn_functions.FoF_graph(df, measure)
#     G.vs["label"]= G.vs["name"]
#     G.vs["degree"]=G.vs.degree()
#     G.vs['betweenness'] = G.betweenness(vertices=None, directed=False, cutoff=None, weights=None, nobigint=True)
#     G.vs['shape']="circle"
    # G.vs["pagerank"]=G.vs.pagerank(directed=False, weights='weight')
#     G.vs["color"] = "rgba(255,0,0,0.2)"
#     return (G)

# if not gramRel == s2n.corpusSelect(cS)[4]:
#     st.title('Cluster 2nd degree by coordination')
#     st.markdown('This method is activated if 1st degree friend network is NOT created from coordination relation. It gives the friend of friend relations by applying coordination. The resulting graph should represent possibilities of other coocurences.')
#     coord_numb= st.slider('Number of collocations in coordination.', 1,50,10)
#     f_connected_by_coordination= f_connected_by_coordination(lemma, pos, corpus, gramRel, s2n.corpusSelect(cS)[4], pos2, limit, coord_numb, measure, harvestSelect)  
#     f_con_by_coordCluster = louvain.find_partition(f_connected_by_coordination, louvain.ModularityVertexPartition)
#     cgcn_functions.clusterAlgoDraw(f_connected_by_coordination, f_con_by_coordCluster, layout, vertexSize, vertexSizeType, vertexLabelSize, vertexLabelSizeType, edgeLabelSize, 'f_by_coordination')    
#     'Clusters in graph', louvain.find_partition(f_connected_by_coordination, louvain.ModularityVertexPartition)
#     if st.checkbox('Visualise 2nd degree by coordination'):
#         fof_image = Image.open('images/f_by_coordination.png')
#         st.image(fof_image, caption='Friends connected by coordination graph for '+lemma+pos
#                 +' with vertices: '+str(FoFgraph.vcount())
#                 +', edges: '+str(FoFgraph.ecount())
#                 +', max degree: '+str(FoFgraph.maxdegree(vertices=None, loops=False))
#                 +', graph density: '+str(FoFgraph.density(loops=False))
#                 +', average_path_length: '+str(FoFgraph.average_path_length())
#                 +', independence: '+str(FoFgraph.alpha())
#                 +', diameter: '+str(FoFgraph.diameter(directed=False, unconn=True, weights=None))
#                 , use_column_width=True)

    

# Pruned version of FoFgraph
st.markdown("<h2 style='text-align:left;font-size:25px; padding-left:20px'><img src='http://emocnet.uniri.hr/wp-content/uploads/2020/09/007-funnel-300x300.png' alt='alt text' width=30>&nbsp;&nbsp;<b>Pruning</b></h2>", unsafe_allow_html=True)
# st.sidebar.subheader('Pruning parameters')
st.sidebar.markdown(read_markdown_file("markdown/pruning_parameters.md"), unsafe_allow_html=True)

# prune by betweenness
btwn = FoFgraph.betweenness()#weights='weight'
ntile_prun=st.sidebar.slider('Betweenness percentile', 0.0,100.0,0.0)
ntile_betweenness = np.percentile(btwn, ntile_prun)
pruned_vs = FoFgraph.vs.select([v for v, b in enumerate(btwn) if b >= ntile_betweenness])
pruned_graph = FoFgraph.subgraph(pruned_vs)

# prune by sbb_importance
pruned_graph.vs['sbb_importance']= cgcn_functions.sbb_importance(pruned_graph)
sbb=pruned_graph.vs['sbb_importance']
sbb_ntile_prun=st.sidebar.slider('Sbb importance percentile', 0.0,100.0,0.0)
ntile_sbb = np.percentile(sbb, sbb_ntile_prun)
pruned_vs = FoFgraph.vs.select([v for v, b in enumerate(sbb) if b >= ntile_sbb])
pruned_graph = FoFgraph.subgraph(pruned_vs)

##Dodano
# prune by sbb_importance2
pruned_graph.vs['sbb2_importance']= cgcn_functions.sbb2_importance(pruned_graph)
sbb2=pruned_graph.vs['sbb2_importance']
sbb2_ntile_prun=st.sidebar.slider('Sbb2 importance percentile', 0.0,100.0,0.0)
ntile_sbb2 = np.percentile(sbb2, sbb2_ntile_prun)
pruned_vs = FoFgraph.vs.select([v for v, b in enumerate(sbb2) if b >= ntile_sbb2])
pruned_graph = FoFgraph.subgraph(pruned_vs)

# prune by sbb3_importance
pruned_graph.vs['sbb3_importance']= cgcn_functions.sbb3_importance(pruned_graph)
sbb3=pruned_graph.vs['sbb3_importance']
sbb3_ntile_prun=st.sidebar.slider('Sbb3 importance percentile', 0.0,100.0,0.0)
ntile_sbb3 = np.percentile(sbb3, sbb3_ntile_prun)
pruned_vs = FoFgraph.vs.select([v for v, b in enumerate(sbb3) if b >= ntile_sbb3])
pruned_graph = FoFgraph.subgraph(pruned_vs)

# prune by shishi_importance
pruned_graph.vs['shishi_importance']= cgcn_functions.shishi_importance(pruned_graph)
shishi=pruned_graph.vs['shishi_importance']
shishi_ntile_prun=st.sidebar.slider('shishi_importance percentile', 0.0,100.0,0.0)
ntile_shishi = np.percentile(shishi, shishi_ntile_prun)
pruned_vs = FoFgraph.vs.select([v for v, b in enumerate(shishi) if b >= ntile_shishi])
pruned_graph = FoFgraph.subgraph(pruned_vs)

# prune by degree
pruned_graph.vs['degree']= pruned_graph.degree()
maxDegree= pruned_graph.maxdegree(vertices=None, loops=False)
prunedDegree = st.sidebar.slider('Degree pruned', 0, maxDegree, (0,maxDegree))
pruned_graph_degree_vs = pruned_graph.vs.select(degree_ge = prunedDegree[0], degree_le= prunedDegree[1])
pruned_graph = FoFgraph.subgraph(pruned_graph_degree_vs)
 



# 'pruned_graph vertices:', len(pruned_graph.vs),'edges:', len(pruned_graph.es), 'diversity', pruned_graph.diversity(), 'degree', pruned_graph.degree()



# Pruned partition type bar sidebox
partitionTypePruned= st.sidebar.selectbox('Partition type for pruned graph', ['mvp', 'cpm'])
if partitionTypePruned== 'mvp':
    cluster_resolution_prun= None
if partitionTypePruned== 'cpm':
    cluster_resolution_prun=st.sidebar.slider('Resolution for pruned graph (0-low, 1-high)', 0.0,1.0,0.5)

clusterAlgo_prun= cgcn_functions.clusterAlgo(pruned_graph, algorithm, partitionTypePruned, cluster_resolution_prun)
with st.beta_expander('Pruned 2nd degree with '+str(len(pruned_graph.vs))+' nodes and '+str(len(pruned_graph.es))+' edges using filters: degree >= '+str(prunedDegree[0])+', betweenness percentile: '+str(ntile_betweenness)+', clustering method: '+str(algorithm)+', partition type: '+str(partitionType)+', resolution: '+str(resolution)):
    # prune by pruned_community
    pruned_DataFrame=cgcn_functions.AlgoDataFrame(pruned_graph, clusterAlgo_prun)
    # 'fof_data', commsDataFrame
    # 'pruned_DataFrame', pruned_DataFrame
    # staviti na pruned_DataFrame svojstvo sbb3_original
    out = (pruned_DataFrame.merge(commsDataFrame[['label', 'sbb3_importance', 'pagerank']], left_on='label', right_on='label'))
        #   .reindex(columns=['id', 'store', 'address', 'warehouse']))
    # 'out',out
    
    comm_pruned_select = st.multiselect('Community number', pruned_DataFrame['commPart'].unique())
    if comm_pruned_select:
        pruned_DataFrame= pruned_DataFrame[pruned_DataFrame['commPart'].isin(comm_pruned_select)]
        'Selected community', pruned_DataFrame
        pruned_graph_comm_vs = pruned_graph.vs.select(label_in = pruned_DataFrame['label'].tolist())
        pruned_graph = pruned_graph.subgraph(pruned_graph_comm_vs)


    config = dict({'scrollZoom': True, 'displaylogo': False, 'displayModeBar': False})
    st.plotly_chart(cgcn_functions.FgraphDraW_plotly_cluster(pruned_graph, clusterAlgo_prun, layout, vertexSize, vertexLabelSize, 'Pruned second degree network', edgeLabelSize),
                    use_container_width=True,config=config)
        # # display pruned image
        # st.image(pruned_image, caption='Pruned graph for '+lemma+pos
        #     +' with vertices: '+str(pruned_graph.vcount())
        #     +', edges: '+str(pruned_graph.ecount())
        #     +', max degree: '+str(pruned_graph.maxdegree(vertices=None, loops=False))
        #     +', graph density: '+str(pruned_graph.density(loops=False))
        #     +', average_path_length: '+str(pruned_graph.average_path_length())
        #     +', independence: '+str(pruned_graph.alpha())
        #     +', diameter: '+str(pruned_graph.diameter(directed=False, unconn=True, weights=None))
        #     , use_column_width=True)



    # Community exploration
    prun_sort_by = st.multiselect('Sort pruned nodes by', pruned_DataFrame.columns)        
    try:    
        st.write(pruned_DataFrame.sort_values(by=prun_sort_by, ascending=False))
    except:
        pass


##############proba boje
# palette = ig.drawing.colors.ClusterColoringPalette(len(clusterAlgo_prun))
# vtransparency = 0.1 # vertex transparency
# pruned_graph.vs['color'] = palette.get_many(clusterAlgo_prun.membership)
# vcolors=[] 
# for v in pruned_graph.vs['color']:
#     vcolor=list(v) #convert tuples to list
#     vcolor[3]= vtransparency #add opacity value
#     vcolors.append(tuple(vcolor)) 
# pruned_graph.vs['color'] = vcolors
# 'boje', [x for x in pruned_graph.vs['color']]
# 'jedna', pruned_graph.vs['color'][2]

# pruned_graph.es['color'] = palette.get_many(clusterAlgo_prun.membership)
# 'Fgraph edge color', pruned_graph.es['color']
# 'neka particija', clusterAlgo_prun[3]
# pruned_graph.vs['membership']=clusterAlgo_prun.membership
# 'clusterAlgo', Fgraph.vs['membership'] #ovo treba dodati u Fgraph ili Fpruned ili tako već
# # edge = (Fgraph.es.select(_within = Fgraph.vs['membership']clusterAlgo_prun[0]))
# 'clustered e'

# edge['color']=vcolors[0]
# 'neka particija', edge['color']


# ##### proba select
# 'pruned_graph', pruned_graph
# 'pruned_graph vs', pruned_graph.vs
# 'clusterAlgo_prun.membership',clusterAlgo_prun.membership
# pruned_graph.vs['membership'] = clusterAlgo_prun.membership
# 'membership', pruned_graph.vs['membership']
# 'select according to membership', pruned_graph.vs.select(membership=1)['membership']
# 'edges selected according to membership', pruned_graph.es(_within =pruned_graph.vs.select(membership=1))['weight']
# 'palette get many', ig.GradientPalette("red", "blue", len(clusterAlgo_prun)).get_many(clusterAlgo_prun.membership)



##################### Find categorical structure with is-a construction
st.markdown("<h2 style='text-align:left;font-size:25px; padding-left:20px'><img src='http://emocnet.uniri.hr/wp-content/uploads/2020/09/032-timeline-300x300.png' alt='alt text' width=30>&nbsp;&nbsp;<b>Labeling<b></h2>", unsafe_allow_html=True)
st.sidebar.markdown(read_markdown_file('markdown/labeling_parameters.md'), unsafe_allow_html=True)


pruned_DataFrame['sbb3_norm']= cgcn_functions.normalize(pruned_DataFrame['sbb3_importance'])
pruned_DataFrame['sbb2_norm']= cgcn_functions.normalize(pruned_DataFrame['sbb2_importance'])

# 'pruned_DataFrame koji gledamo', pruned_DataFrame[['label', 'commPart', 'sbb3_norm','sbb3_importance','sbb2_norm','shishi_importance', 'degree', 'betweenness', 'weighted', 'pagerank']]

def is_a_F(lemposSeries):
    df=pd.DataFrame()
    for row in lemposSeries:
        try:
            df_friend= cgcn_functions.friends_df(language, row.split('-')[0], '-'+row.split('-')[1], corpus, corpusID, '"%w" is a ...', '-n', 25, measure, harvestSelect, 'directed')
            df= df.append(df_friend)
        except:
            pass        
    return df


#get number of communities
comms = pruned_DataFrame['commPart'].unique()
#select how to sort the community lexemes in is_a graphs 
sort_is_a= st.sidebar.selectbox('Sort is_a dataset by ', ['weighted_degree','pagerank','sbb3_isa', 'eigen_c',   'betweenness', 'degree','label'])
# 'sort_is_a', sort_is_a
# In progress - create a function
# def get_is_a(pruned_DataFrame, sort_is_a):
#     'from communities extract labels'
#     comms = pruned_DataFrame['commPart'].unique()
#     counter=-1
#     for comm in comms:
#         counter=counter+1
#         items= pruned_DataFrame[pruned_DataFrame['commPart']==comm]['label']
#         is_a_net = is_a_F(items)
#         is_a_net_network= cgcn_functions.Fgraph(is_a_net, measure)

# number of labels    
label_num=st.sidebar.slider('Choose number of is_a labels', 1,10,2, key='labels_number')
# display community members
ch_is_a_comm_members=st.sidebar.checkbox('Display community members', value=True)
ch_is_a_network_df=st.sidebar.checkbox('Display first degree is_a dataframe', value=True)
ch_is_a_network_graph=st.sidebar.checkbox('Display first degree is_a network', value=False)
ch_is_a_network_second=st.sidebar.checkbox('Display second degree is_a dataframe', value=False)
ch_is_a_wordnet_hyper=st.sidebar.checkbox('Display wordnet hypernym results', value=True)
ch_is_a_wordnet_domains=st.sidebar.checkbox('Display wodnet domains for the community', value=True)

counter=-1
for comm in comms:
    counter=counter+1
    # 'counter', counter
    # members of a class 
    items= pruned_DataFrame[pruned_DataFrame['commPart']==comm]['label']
    # for items in a community get first degree collocates
    is_a_net= is_a_F(items)
    # 'vidi is_a_F df za comm', comm, is_a_net
    try:
        is_a_net_network = cgcn_functions.Fgraph_is_a(is_a_net, measure)
    except:
        'is_a_net_network in not computed'
# create Friend graph
    try:
        is_a_net_df = pd.DataFrame({attr: is_a_net_network.vs[attr] for attr in is_a_net_network.vertex_attributes()}).sort_values(by = sort_is_a, ascending=False).drop(['color'], axis=1)
        # 'is_a_net_df', is_a_net_df
        # is_a_net_network.vs['sbb3_isa']= cgcn_functions.sbb3_importance(is_a_net_network)  
        # 'is_a_net_network_sbb3', is_a_net_network.vs['sbb3_isa']
        # 'is_a_net_network',is_a_net_network
        # create dataframe from graph
        # 'is_a_net_df', is_a_net_df
    except:
        'try: is_a_net_df is not computed for:'
        st.markdown('<p> Associated members of a <b>class '+str(counter)+"</b>: <i>"+', '.join(items.values)+'</i></p>', unsafe_allow_html=True)
        pass


    with st.beta_expander('Sense '+str(counter+1)+': '+lempos+" is related to "+", ".join(is_a_net_df.head()['label'][0:label_num]).upper()):
        # Display label results
        # st.markdown("<h3 style='background-color:aliceblue'><i>&nbsp;&nbsp;"+lempos+"</i> is related to "+", ".join(is_a_net_df.head()['label'][0:label_num]).upper()+"</h3>", unsafe_allow_html=True)    
        if ch_is_a_comm_members:
            st.markdown('<p>'+str(len(items))+' associated members of a <b>class '+str(counter)+"</b>: <i>"+', '.join(items.values)+'</i></p>', unsafe_allow_html=True) 
        if ch_is_a_network_df:
            st.markdown('<p>First degree nodes of <b>class '+str(counter)+'</b> <i>is_a ... label</i> network:</p>', unsafe_allow_html=True)
            st.dataframe(is_a_net_df)
        if ch_is_a_network_graph:
            st.markdown('<p>First degree network of <b>class '+str(counter)+'</b> <i>is_a ... label</i>:</p>', unsafe_allow_html=True)
            is_a_net_network_Fgraph=cgcn_functions.FgraphDraW_plotly(is_a_net_network, layout, vertexSize, vertexLabelSize, 'is_a_graph_plotly_pruned'+str(counter), edgeLabelSize)
            st.plotly_chart(is_a_net_network_Fgraph, use_container_width=True)
        if ch_is_a_network_second:
            # get second degree is_a collocations
            is_a_net2 =  is_a_F(is_a_net_df['label'])
            # create second degree is_a graph
            is_a_net2_network= cgcn_functions.Fgraph(is_a_net2, measure)
            # create dataframe for second degree
            is_a_net2_df= cgcn_functions.df_from_graph(is_a_net2_network)
            st.markdown('<p>Second degree nodes of <b>class '+str(counter)+'</b> <i>is_a ...label is_a...label</i> network:</p>', unsafe_allow_html=True)
            st.dataframe(is_a_net2_df.sort_values(by=sort_is_a, ascending=False))

        #wordnet label calculation
        if ch_is_a_wordnet_hyper:
            wn_hypernyms= wordnet_functions.get_hypernyms(items, language)#(is_a_net_df['label'])
            wn_hypernyms['weight']=1

            sbb_mapa =  commsDataFrame[commsDataFrame['label'].isin(wn_hypernyms['lempos'])][['label', 'sbb3_importance']] #izlučivanje sbb vrijednosti iz FoF na temelju label 
            wn_hypernyms = wn_hypernyms.merge(sbb_mapa, left_on='lempos', right_on='label')[['source','target', 'weight', 'sbb3_importance']]#stavlja sbb na temelju lemposa

            'vidi wn_hypernyms za staviti sbb na weight', wn_hypernyms
            #create tuples from wn_hypernyms.values
            tuples = [tuple(x) for x in wn_hypernyms.values]
            # #create igraph object from tuples
            G=ig.Graph.TupleList(tuples, directed = True, edge_attrs=['weight', 'sbb3_importance'], vertex_name_attr='name', weights=False)
            #create vertex labels from name attr
            G.vs["syn"]= [str(x)[8:-2] for x in G.vs["name"]]
            G.vs["label"]= G.vs["name"]
            G.vs["definition"]=[wordnet_functions.get_definition(x) for x in G.vs["syn"]]
            G.vs["degree"]=G.vs.degree()
            G.vs["pagerank"]=G.vs.pagerank(directed=True, weights='sbb3_importance')
            # G.vs["personalized_pagerank"]= G.vs.personalized_pagerank(directed=True, weights='weight')
            G.vs["betweenness"] = G.betweenness()
            # try:
            G.vs["weighted_degree"] = G.strength([i.index for i in G.vs], weights='sbb3_importance', mode='IN')
            # except:
            #     pass
            # G.vs['eigen_c'] = G.eigenvector_centrality(directed=False, scale=True, weights= 'weight', return_eigenvalue=False)
            try:
                G.vs['sbb3'] = cgcn_functions.sbb3_importance(G)
            except:
                pass
            wndfg= pd.DataFrame({attr: G.vs[attr] for attr in G.vertex_attributes()}).drop(['name'], axis=1)
            wndfg= wndfg.sort_values(by='weighted_degree', ascending=False)
            
            # ovo je problem zašto ne želi procesirati sbb_importance - G.vs['label'] su synsetovi a ne stringovi
            # wndfg['bss_importance']= cgcn_functions.sbb_importance(G)
            
            st.markdown("<p style='background-color:aliceblue; padding-left:5px'>Wordnet <i>hypernym synsets</i> for <b>class "+str(counter)+"</b>: "+str(', '.join(wndfg['syn'].astype(str).str.slice(start=0,stop=-5)[0:int(label_num)]).upper())+"</p>", unsafe_allow_html=True) 
            '', wndfg#.head()
            valency=wordnet_functions.get_valency_for_lempos(wndfg["syn"])
            st.markdown("<p style='background-color:PapayaWhip; padding-left:5px'>Wordnet <i>valency</i> for synsets of <b>class"+str(counter)+"</b>: "+ json.dumps(valency).replace('"','').replace('{', '').replace('}', ''), unsafe_allow_html=True)
            if st.checkbox('Wordnet hypernyms for words in class '+str(counter)):
                st.dataframe(wn_hypernyms) #.groupby('target').count()['source'].sort_values(ascending=False)

            if ch_is_a_wordnet_domains:
                tuples_d=[]
                domainsList=[]
                G = ig.Graph()
                # get synwords from items using gethypernyms 
                syns_words=wordnet_functions.get_synset_from_lempos(items, language).astype(str).str.slice(start=8, stop=-7)
                for word in syns_words.tolist():
                    try:
                        doms= spwf.get_domains_for_word(word)
                        for dom in doms:
                            tuples_d.append((word, dom, 1))
                            domainsList.append(dom)
                    except:
                        pass
                domSeries = pd.Series(domainsList)
                domSeries_count = domSeries.value_counts().rename_axis('domains').reset_index(name='counts')
                # calculate the proportion of the domain for class
                domSeries_count['proportion']= [x/domSeries.count() for x in domSeries_count['counts']]
                
                'Wordnet *domains* for **class '+str(counter)+'**:', domSeries_count       
                if st.checkbox('Wordnet domains dataset and graph for class'+str(counter)):
                    G=ig.Graph.TupleList(tuples_d, directed=True, edge_attrs=['weight'], vertex_name_attr='name', weights=False)
                    G.vs["label"]=G.vs["name"]
                    G.vs["color"]="rgba(255,0,0,0.2)"
                    G.vs["degree"]=G.vs.degree()
                    G.vs["pagerank"]=G.vs.pagerank(directed=True)
                    G.vs["betweenness"] = G.betweenness()
                    # G.vs['eigen_c'] = G.eigenvector_centrality(directed=True, scale=True, weights= 'weight', return_eigenvalue=False)
                    wndom= pd.DataFrame()
                    wndom["name"]= [x for x in G.vs["name"]]
                    # wndom['eigen_c']=G.vs['eigen_c']
                    wndom["degree"]=G.vs["degree"]
                    wndom["pagerank"]=G.vs["pagerank"]
                    'Word domain graph', wndom.sort_values(by='degree', ascending=False)
                    wndom_graph=cgcn_functions.FgraphDraW_plotly(G, layout, vertexSize, vertexLabelSize, 'domain_graph_plotly_pruned'+str(counter), edgeLabelSize)
                    st.plotly_chart(wndom_graph)
                
        st.markdown('----')
##################### compare two graphs





################################################ find path
# st.title('Path finding algorithms')
# pathSource= st.text_input('Select a source lexeme from the network', lemma+pos)#, commsDataFrame['label'])
# pathTarget = st.text_input('Write lempos value of a Target lexeme')
# kPaths = st.number_input('How many paths?', 2)
# pathDirection = st.selectbox('Select direction of the path', ('IN', 'OUT', 'BOTH'))


# pathResult = cgcn_functions.findPath(pathSource, pathTarget, kPaths,gramRel, pathDirection, corpusID).to_data_frame()
# 'K_ShortestPaths in', corpusID, ': ', pathSource, '--[',  gramRel, ']--', pathTarget
# for index, row in pathResult.iterrows():
#     st.write(row)

# posMiddle=st.text_input('Choose your connecting POS (-n, -v, -j, -r, etc..)')
# def findMiddle(pathSource, pathTarget, posMiddle, pathDirection, corpusID):
#     q='''
#     MATCH p= (start:Lemma{lempos:$pathSource})--> (middle:Lemma{lpos:$posMiddle})--> (end:Lemma{lempos:$pathTarget}) 
#     return middle.lempos
#     '''
#     df= graph.run(q, pathSource=pathSource, pathTarget=pathTarget, posMiddle=posMiddle, pathDirection=pathDirection, corpusID=corpusID )
#     return (df)
# connecting=findMiddle(pathSource, pathTarget, posMiddle, pathDirection, corpusID)
# if connecting:
#     'Connecting lexemes', connecting.to_data_frame().drop_duplicates()



# EndTime
endTime= time.time()
'ConGraCNet calculations finished in', endTime-startTime

# To do
# napraviti da se samo koordinacija izračunava kao neusmjereni graf a svi ostali kao usmjereni