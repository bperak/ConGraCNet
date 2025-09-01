# -*- coding: utf-8 -*-
# Imports
import authSettings as aS
import sketch2Neo2 as s2n
from py2neo import Graph
import cgcn_functions
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

# Title
st.markdown(read_markdown_file("markdown/introduction.md"), unsafe_allow_html=True)

# Sidebar logo
st.sidebar.markdown(read_markdown_file("markdown/corpus_selection.md"), unsafe_allow_html=True)
# Sidebar Corpus select
cS= st.sidebar.selectbox('Corpus', s2n.corpusListDf['cS'], key='corpus')


# Sidebar Corpus data based on the selection
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

# Sidebar Lemma pos select
st.sidebar.markdown(read_markdown_file("markdown/lemma_selection.md"), unsafe_allow_html=True)
try:
    lemma = st.sidebar.text_input('Lemma', initial_lexeme,  key='lemma')
    pos = '-'+st.sidebar.text_input('Part of speech',  initial_pos, key='pos')
    lempos = lemma+pos
except:
    'No data for this lemma+pos'
    pass


####### Todo WIP - corpus info from api
# st.title('Corpus info')
# '', s2n.corpus_info(corpSelect[3], aS.userName, aS.apiKey)

##########################################################
# Source lexeme
st.markdown("<h2 style='background-color:rgb(255, 153, 51);color:white;text-align:lefttext-align:left;padding: 10px 20px 10px 20px'><img src='http://emocnet.uniri.hr/wp-content/uploads/2020/09/017-distribution-300x300.png' alt='alt text' width=30>&nbsp;&nbsp;Source lexeme</h2>", unsafe_allow_html=True)
# st.title('Source lexeme')
'##', lempos
'Corpus: ', corpSelect[0] 

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
    'Frequency in corpus: ', source_lema_data['freq'].values[0]
    'Relative frequency in corpus (per million): ', source_lema_data['relFreq'].values[0]
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


# GramRel list - for a lemma in a corpus give a list of gramRels and count of collocation
listGramRels= cgcn_functions.lemmaGramRels(lemma, pos, corpusID, language)   
# GramRel list option
listGramRelsCheck =st.checkbox('Grammatical relations list for '+lemma+pos+':', key= 'listGramRels', value=True) # Checkbox for gramrel list
if listGramRelsCheck:
    
    # '', listGramRels.sort_values(by='count', ascending=False)
     
    fig = px.bar(listGramRels.sort_values(by='count', ascending=True), 
                x="count", y="gramRels", orientation='h', opacity=0.7, log_x=False, height=600, 
                color='count', color_continuous_scale="blugrn",
                text='count')
    fig.update_layout(
    paper_bgcolor = 'rgba(0,0,0,0)', plot_bgcolor = 'rgba(240,240,240,0.5)',
    font_family="Verdana",
    font_color="black",
    font_size=10,
    title_font_family="Verdana",
    title_font_color="red",
    legend_title_font_color="green",
    coloraxis_showscale=False)
    fig.update_xaxes(title_font_family="Arial")
    st.plotly_chart(fig)


# Lemma network profile - for a lemma in a corpus gives a collocation profile with all gramRels
lemmaProfile=st.checkbox('Collocation list for '+lemma+pos+':', key= 'lemmaProfile')
if lemmaProfile:
    df= cgcn_functions.lemmaGramRelProfile(lemma, pos, corpusID, language)
    '', df.dropna()


# Set minimal frequency and score parameters
minParameters=st.sidebar.checkbox('Set minimal frequency and score')
if minParameters:
    scoreMin=st.number_input('Minimum score', 0)
    freqMin=st.number_input('Minimum frequency', 0)

####### Wordnet integration
if st.checkbox('Wordnet profile', value=True, key='wordnet_profile'):
    wn_hypernym=wordnet_functions.get_hypernyms([lempos], language)
    wn_hypernym['s_definition']= [wordnet_functions.get_definition(str(x)[8:-2]) for x in wn_hypernym['source']]
    wn_hypernym['t_definition']= [wordnet_functions.get_definition(str(x)[8:-2]) for x in wn_hypernym['target']]
    '*Synsets, definitions* and *hypernyms* for ***'+lempos+'***:'#, wn_hypernym #wordnet_functions.get_hypernyms([lempos], language)
    counter=0
    for index,row in wn_hypernym.iterrows():
        counter=counter+1
        st.markdown(str(counter)+': ***'+str(row['source'])[8:-2]+'*** defined as *'+str(row['s_definition'])+'* with hypernym ***'+str(row['target'])[8:-2]+'*** defined as *'+str(row['t_definition'])+'*')
    unique_syns_words=wn_hypernym['source'].astype(str).str.slice(start=8, stop=-7).unique()# unique words
    
    for syn_word in unique_syns_words:
        syn_domains= ', '.join(spwf.get_domains_for_word(syn_word))
        st.markdown('*Domains* for ***'+syn_word+'***: '+syn_domains)
        '*Wordnet valency* for synset word ***'+syn_word+'***: '+ json.dumps(wordnet_functions.get_valency_for_lempos([syn_word])).replace('"','').replace('{', '').replace('}', '')



##################### Network construction (lemma, pos, gramRel, measure, pos2)
st.markdown("<h2 style='background-color:rgb(255, 102, 0);color:white;text-align:left; padding:10px 20px 10px 20px'><img src='http://emocnet.uniri.hr/wp-content/uploads/2020/09/020-algorithm-300x300.png' alt='alt text' width=30>&nbsp;&nbsp;Network construction</h2>", unsafe_allow_html=True)
st.write('## First degree (Source-Friend)')
# GramRel selection (selectbox, selection based on the listGramRels df, initial is set from the corpSelect[4])
st.sidebar.markdown(read_markdown_file("markdown/grammar_parameters.md"), unsafe_allow_html=True)
# st.sidebar.subheader('Grammar relation parameters')
gramRel= st.sidebar.selectbox('Choose grammatical relation', listGramRels['gramRels'], list(listGramRels['gramRels']).index(corpSelect[4]))

def direction(gramRel, initial):
    'define direction of the graph based on the gramRel'
    if gramRel == initial:
        direction = 'undirected'
    else:
        direction = 'directed'
    return (direction)


# GramReldf (function)
gramReldf= cgcn_functions.lemmaByGramRel(language, lemma, pos, gramRel, corpusID, measure)

# Filter pos2 (selectbox from gramReldf)
pos2='-'+st.sidebar.selectbox('Filter the pos value of the friend lexeme', gramReldf['friend'].str[-1].unique() )


# activate function friends_df() and store as friendCoocurences variable
friendCoocurences = cgcn_functions.friends_df(language, lemma, pos, corpus, corpusID, gramRel, pos2, limit, measure, harvestSelect, direction(gramRel,corpSelect[4]))

# List dataframe friends
'', len(friendCoocurences),' collocations of ', str(lemma+pos), ' in ',  gramRel, ", ordered by: ", measure, " in ", corpus, ', direction: ', direction(gramRel,corpSelect[4] )
if st.checkbox('Friends dataset', value=False):
    '',friendCoocurences

# Draw Friends Score vs Frequency Rank
if st.checkbox('Friends rank scatterplot', value=True):
    cgcn_functions.drawFriendCoocurences(friendCoocurences)

#calculate friend graph
Fgraph = cgcn_functions.Fgraph(friendCoocurences, measure) 

# Draw Friend Igraph
if st.checkbox('Visualize Source-Friend graph'):
    cgcn_functions.FgraphDraw(Fgraph, 'fr')
    #represent the Fgraph image with caption
    Fgraph_image = Image.open('images/Fgraph.png')
    st.image(Fgraph_image, caption='Friends graph for '+lemma+pos
        +' with vertices: '+str(Fgraph.vcount())
        +', edges: '+str(Fgraph.ecount())
        +', graph density: '+str(Fgraph.density(loops=False))
        +', diameter: '+str(Fgraph.diameter(directed=False, unconn=True, weights=None))
        ,use_column_width=True)


# Friend-of-friend network
st.write('## Second degree (Source-Friend-Friend)')

# Store Friend-of-friend network dataframe in a df_fof variable
df_fof = cgcn_functions.FoFData(language, friendCoocurences, lemma, pos, corpus, corpusID, gramRel, pos2, limit, measure, harvestSelect, direction(gramRel, corpSelect[4]))
len(df_fof),' collocations of *'+lemma+pos+"* connected with *"+gramRel+"* in the first degree and *"+gramRel+"* in the second degree"
if st.checkbox('Second degree (FoF) dataset'):
    '', df_fof #[['source','friend', 'freq', 'score', 'gramRel']]

#store the result in a FoFgraph variable
FoFgraph=cgcn_functions.FoF_graph(df_fof, measure)

# st.subheader('FoF graph visualization')
####################################################### Visualization selection

# st.sidebar.subheader('Visualization parameters')
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


# Edge label size
edgeLabelSize = st.sidebar.slider('Edge label size', 0.0,10.0,1.0)

# Layout selection
layout= st.sidebar.selectbox('Layout type', ('fr', 'kk', 'lgl', 'circle', 'drl', 'random', 'tree'))
# more layouts to come #####################

# visualize network
if st.checkbox('FoF graph visualization'):
    cgcn_functions.FoFgraphDraw(FoFgraph, layout, vertexSize, vertexSizeType, vertexLabelSize, vertexLabelSizeType)
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
st.markdown("<h2 style='background-color:mediumpurple;color:white;text-align:lefttext-align:left; padding:10px 20px 10px 20px'><img src='http://emocnet.uniri.hr/wp-content/uploads/2020/09/015-diagram-300x300.png' alt='alt text' width=30>&nbsp;&nbsp;Clustering</h2>", unsafe_allow_html=True)
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
'Clustering second degree network yielded', len(clusterAlgo), 'communities using algorithm: *'+algorithm+'*, partition: *'+partitionType+'*, resolution: *'+str(resolution)+'*.'
if st.checkbox('Second degree (FoF) clustered network list'):
    '', clusterAlgo

# cluster visualization
if st.checkbox('Second degree (FoF) clustered network graph'):
    # activate visualisation function clusterAlgoDraw
    cgcn_functions.clusterAlgoDraw(FoFgraph, clusterAlgo, layout, vertexSize, vertexSizeType, vertexLabelSize, vertexLabelSizeType, 2.0, 'FoFClusterAlgo') 
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




# Explore communities
st.subheader('Community exploration')
commsDataFrame=cgcn_functions.AlgoDataFrame(FoFgraph, clusterAlgo)
algo_sort_by = st.multiselect('Sort nodes by', commsDataFrame.columns)
try:    
    st.write(commsDataFrame.sort_values(by=algo_sort_by, ascending=False))
except:
    pass

##### 3 d visualisation using the # https://github.com/vasturiano/3d-force-graph 
# create json data of nodes, groups and links
json_data =cgcn_functions.json_FoF_clustered(df_fof, commsDataFrame, measure)
# 'Proba json', json_data 
# Write json data in the datasets folder - this method is not ok for production due to the writing - the data should be stored in memory
out_file=open('3d-force-graph/example/datasets/border.json', 'w')
out_file.write(json_data)
out_file.close()

'Vizualizacija json 3d network'
# stranica se mora vrtiti 
components.iframe('http://127.0.0.1:5501/3d-force-graph/example/tree/index.html', height=500)


proba_html='''
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    <div id="accordion">
      <div class="card">
        <div class="card-header" id="headingOne">
          <h5 class="mb-0">
            <button class="btn btn-link" data-toggle="collapse" data-target="#collapseOne" aria-expanded="true" aria-controls="collapseOne">
            Collapsible Group Item #1
            </button>
          </h5>
        </div>
        <div id="collapseOne" class="collapse show" aria-labelledby="headingOne" data-parent="#accordion">
          <div class="card-body">
            Collapsible Group Item #1 content
          </div>
        </div>
      </div>
      <div class="card">
        <div class="card-header" id="headingTwo">
          <h5 class="mb-0">
            <button class="btn btn-link collapsed" data-toggle="collapse" data-target="#collapseTwo" aria-expanded="false" aria-controls="collapseTwo">
            Collapsible Group Item #2
            </button>
          </h5>
        </div>
        <div id="collapseTwo" class="collapse" aria-labelledby="headingTwo" data-parent="#accordion">
          <div class="card-body">
            Collapsible Group Item #2 content
          </div>
        </div>
      </div>
    </div>
'''
components.html(proba_html)

#################### cluster 2nd degree friends by coordination
# identify source-friends network using a grammatical relation (not coordination) and cluster friends using f_graph with relation=coordination
def f_connected_by_coordination(lemma, pos, corpus, gramRel, initial, pos2, limit, coord_numb, measure, harvestSelect):
    # get friends df
    df = cgcn_functions.friends_df(language, lemma, pos, corpus, corpusID, gramRel, pos2, limit, measure, harvestSelect, direction(gramRel, corpSelect[4]))[['source', 'friend', 'freq','score','gramRel','corpus']]
    # get friend_of_friends_by_coordination
    try:            
        for index, row in df.iterrows():
            df2= cgcn_functions.friends_df(language, row['friend'][0:-2], row['friend'][-2:], corpus, corpusID, initial, pos2, coord_numb, measure, harvestSelect, direction(gramRel, initial))        
            #append the dataframe of df2 to df
            df2= df2.rename(columns={'friend': 'source', 'source':'friend'})
            df=df.append(df2, sort=False)
            df=df.drop_duplicates().dropna()   
    except:
        pass
    
    # create a FoF graph
    G= cgcn_functions.FoF_graph(df, measure)
    G.vs["label"]= G.vs["name"]
    G.vs["degree"]=G.vs.degree()
    G.vs['betweenness'] = G.betweenness(vertices=None, directed=False, cutoff=None, weights=None, nobigint=True)
    G.vs['shape']="circle"
    G.vs["pagerank"]=G.vs.pagerank(directed=False, weights='weight')
    G.vs["color"] = "rgba(255,0,0,0.2)"
    return (G)

if not gramRel == s2n.corpusSelect(cS)[4]:
    st.title('Cluster 2nd degree by coordination')
    st.markdown('This method is activated if 1st degree friend network is NOT created from coordination relation. It gives the friend of friend relations by applying coordination. The resulting graph should represent possibilities of other coocurences.')
    coord_numb= st.slider('Number of collocations in coordination.', 1,50,10)
    f_connected_by_coordination= f_connected_by_coordination(lemma, pos, corpus, gramRel, s2n.corpusSelect(cS)[4], pos2, limit, coord_numb, measure, harvestSelect)  
    f_con_by_coordCluster = louvain.find_partition(f_connected_by_coordination, louvain.ModularityVertexPartition)
    cgcn_functions.clusterAlgoDraw(f_connected_by_coordination, f_con_by_coordCluster, layout, vertexSize, vertexSizeType, vertexLabelSize, vertexLabelSizeType, edgeLabelSize, 'f_by_coordination')    
    'Clusters in graph', louvain.find_partition(f_connected_by_coordination, louvain.ModularityVertexPartition)
    if st.checkbox('Visualise 2nd degree by coordination'):
        fof_image = Image.open('images/f_by_coordination.png')
        st.image(fof_image, caption='Friends connected by coordination graph for '+lemma+pos
                +' with vertices: '+str(FoFgraph.vcount())
                +', edges: '+str(FoFgraph.ecount())
                +', max degree: '+str(FoFgraph.maxdegree(vertices=None, loops=False))
                +', graph density: '+str(FoFgraph.density(loops=False))
                +', average_path_length: '+str(FoFgraph.average_path_length())
                +', independence: '+str(FoFgraph.alpha())
                +', diameter: '+str(FoFgraph.diameter(directed=False, unconn=True, weights=None))
                , use_column_width=True)

    

# Pruned version of FoFgraph
st.markdown("<h2 style='background-color:indigo;color:white;text-align:lefttext-align:left; padding:10px 20px 10px 20px'><img src='http://emocnet.uniri.hr/wp-content/uploads/2020/09/007-funnel-300x300.png' alt='alt text' width=30>&nbsp;&nbsp;Pruning</h2>", unsafe_allow_html=True)
# st.sidebar.subheader('Pruning parameters')
st.sidebar.markdown(read_markdown_file("markdown/pruning_parameters.md"), unsafe_allow_html=True)
#betweenness parameter
btwn = FoFgraph.betweenness()#weights='weight'
ntile_prun=st.sidebar.slider('Betweenness percentile', 0.1,100.0,50.0)
ntile_betweenness = np.percentile(btwn, ntile_prun)
pruned_vs = FoFgraph.vs.select([v for v, b in enumerate(btwn) if b >= ntile_betweenness])
pruned_graph = FoFgraph.subgraph(pruned_vs)
# 'pruned_graph', type(pruned_graph)
# 'pruned_graph vertices:', len(pruned_graph.vs),'edges:', len(pruned_graph.es), 'diversity', pruned_graph.diversity(), 'degree', pruned_graph.degree()

# prune by degree
pruned_graph.vs['degree']= pruned_graph.degree()
maxDegree= pruned_graph.maxdegree(vertices=None, loops=False)
prunedDegree = st.sidebar.slider('Degree pruned', 0, maxDegree, (2,maxDegree))
pruned_graph_degree_vs = pruned_graph.vs.select(degree_ge = prunedDegree[0], degree_le= prunedDegree[1])
pruned_graph = FoFgraph.subgraph(pruned_graph_degree_vs)

# Pruned partition type bar sidebox
partitionTypePruned= st.sidebar.selectbox('Partition type for pruned graph', ['mvp', 'cpm'])
if partitionTypePruned== 'mvp':
    cluster_resolution_prun= None
if partitionTypePruned== 'cpm':
    cluster_resolution_prun=st.sidebar.slider('Resolution for pruned graph (0-low, 1-high)', 0.0,1.0,0.5)

clusterAlgo_prun= cgcn_functions.clusterAlgo(pruned_graph, algorithm, partitionTypePruned, cluster_resolution_prun)
'Pruning second degree (FoF) clustered network with *degree* >= '+str(prunedDegree[0])+', *betweenness percentile*: '+str(ntile_betweenness)+', *clustering method*: '+str(algorithm)+', *partition type*: '+str(partitionType)+', *resolution*: '+str(resolution), 
'', clusterAlgo_prun

# prune by pruned_community
pruned_DataFrame=cgcn_functions.AlgoDataFrame(pruned_graph, clusterAlgo_prun)
comm_pruned_select = st.multiselect('Community number', pruned_DataFrame['commPart'].unique())
if comm_pruned_select:
    pruned_DataFrame= pruned_DataFrame[pruned_DataFrame['commPart'].isin(comm_pruned_select)]
    'Selected community', pruned_DataFrame
    pruned_graph_comm_vs = pruned_graph.vs.select(label_in = pruned_DataFrame['label'].tolist())
    pruned_graph = pruned_graph.subgraph(pruned_graph_comm_vs)

if limit < 20:
    graphVisualise= True
else:
    graphVisualise= False
if st.checkbox('Pruned second degree (FoF) clustered graph', value= graphVisualise):
    cgcn_functions.clusterAlgoDraw(pruned_graph, clusterAlgo_prun, layout, vertexSize, vertexSizeType, vertexLabelSize, vertexLabelSizeType, edgeLabelSize, 'pruned')
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
# if st.checkbox('Visualize pruned graph with Plotly'): 
#     plotly_Fgraph=cgcn_functions.FgraphDraW_plotly(pruned_graph, layout, vertexSize, vertexLabelSize, 'F_graph_plotly_pruned', edgeLabelSize)
#     st.plotly_chart(plotly_Fgraph)

# TODO!! # Plotly graphing function with clusters  
# plotly_Fgraph_cluster=cgcn_functions.FgraphDraW_plotly_cluster(pruned_graph, clusterAlgo_prun, layout, vertexSize_prun, vertexLabelSize_prun, 'F_graph_plotly_pruned', edgeLabelSize_prun)
# st.plotly_chart(plotly_Fgraph_cluster)

# Community exploration
prun_sort_by = st.multiselect('Sort pruned nodes by', pruned_DataFrame.columns)        
try:    
    st.write(pruned_DataFrame.sort_values(by=prun_sort_by, ascending=False))
except:
    pass
##################### Find categorical structure with is-a construction
st.markdown("<h2 style='background-color:royalblue;color:white;text-align:lefttext-align:left; padding:10px 20px 10px 20px'><img src='http://emocnet.uniri.hr/wp-content/uploads/2020/09/032-timeline-300x300.png' alt='alt text' width=30>&nbsp;&nbsp;Labeling</h2>", unsafe_allow_html=True)
st.sidebar.markdown(read_markdown_file('markdown/labeling_parameters.md'), unsafe_allow_html=True)
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
sort_is_a= st.sidebar.selectbox('Sort is_a dataset by ', ['eigen_c', 'pagerank', 'weighted_degree', 'betweenness', 'degree','label'])

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
    # members of a class 
    items= pruned_DataFrame[pruned_DataFrame['commPart']==comm]['label']
    # for items in a community get first degree collocates
    is_a_net= is_a_F(items)
    # create Friend graph
    try:
        is_a_net_network = cgcn_functions.Fgraph(is_a_net, measure)
        # create dataframe from graph
        is_a_net_df=cgcn_functions.df_from_graph(is_a_net_network).sort_values(by=sort_is_a, ascending=False)
        # Display label results
        st.markdown("<h3 style='background-color:aliceblue'><i>&nbsp;&nbsp;"+lempos+"</i> is related to "+", ".join(is_a_net_df.head()['label'][0:label_num]).upper()+"</h3>", unsafe_allow_html=True)    
        if ch_is_a_comm_members:
            st.markdown('<p>'+str(len(items))+' associated members of a <b>class '+str(counter)+"</b>: <i>"+', '.join(items.values)+'</i></p>', unsafe_allow_html=True) 
        if ch_is_a_network_df:
            st.markdown('<p>First degree nodes of <b>class '+str(counter)+'</b> <i>is_a ... label</i> network:</p>', unsafe_allow_html=True)
            st.dataframe(is_a_net_df.head())
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
    except:
        st.markdown('<p> Associated members of a <b>class '+str(counter)+"</b>: <i>"+', '.join(items.values)+'</i></p>', unsafe_allow_html=True)
        pass

    #wordnet label calculation
    if ch_is_a_wordnet_hyper:
        wn_hypernyms= wordnet_functions.get_hypernyms(items, language)#(is_a_net_df['label'])
        wn_hypernyms['weight']=1
        #create tuples from wn_hypernyms.values
        tuples = [tuple(x) for x in wn_hypernyms.values]
        # #create igraph object from tuples
        G=ig.Graph.TupleList(tuples, directed = True, edge_attrs=['weight'], vertex_name_attr='name', weights=False)
        #create vertex labels from name attr
        G.vs["label"]= G.vs["name"]
        G.vs["degree"]=G.vs.degree()
        G.vs["pagerank"]=G.vs.pagerank(directed=False, weights='weight')
        G.vs["personalized_pagerank"]= G.vs.personalized_pagerank(directed=False, weights='weight')
        # G.vs["weighted_degree"] = G.strength(G.vs["label"], weights='weight', mode='IN')
        G.vs["betweenness"] = G.betweenness()
        G.vs['eigen_c'] = G.eigenvector_centrality(directed=False, scale=True, weights= 'weight', return_eigenvalue=False)
        wndfg= pd.DataFrame()
        # wndfg["label"]=G.vs["label"]
        wndfg["syn"]= [str(x)[8:-2] for x in G.vs["label"]]
        wndfg["definition"]=[wordnet_functions.get_definition(x) for x in wndfg["syn"]]
        wndfg['eigen_c']=G.vs['eigen_c']
        wndfg["degree"]=G.vs["degree"]
        wndfg["pagerank"]=G.vs["pagerank"]
        wndfg= wndfg.sort_values(by='pagerank', ascending=False)
        st.markdown("<p style='background-color:aliceblue; padding-left:5px'>Wordnet <i>hypernym synsets</i> for <b>class "+str(counter)+"</b>: "+str(', '.join(wndfg['syn'].astype(str).str.slice(start=0,stop=-5)[0:int(label_num)]).upper())+"</p>", unsafe_allow_html=True) 
        '', wndfg.head()
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
                G.vs['eigen_c'] = G.eigenvector_centrality(directed=True, scale=True, weights= 'weight', return_eigenvalue=False)
                wndom= pd.DataFrame()
                wndom["name"]= [x for x in G.vs["name"]]
                wndom['eigen_c']=G.vs['eigen_c']
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