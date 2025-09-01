# -*- coding: utf-8 -*-


'''
23.12.2021
Ovo je verzija cgcn streamlit appa u kojoj je:
1) ubačena mjera sli za izračunavanje centralnosti čvorova
2) na temelju sli se radi ADV, a može se i postaviti za određivanje GSV 
'''
# Imports
from operator import mod

from streamlit.proto.Markdown_pb2 import Markdown
import authSettings as aS
import sketch2Neo2 as s2n
from py2neo import Graph
import cgcn_functions_3_6 as cgcn_functions
import wordnet_functions
import spacy_wordnet_functions as spwf
import sentiment_functions_3_6 as sentiment_functions 
import math
import time
import pandas as pd
pd.set_option('display.max_colwidth', 0)
pd.set_option("precision", 5)
import json
import matplotlib
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
# from st_aggrid import AgGrid

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

# Start time
startTime= time.time()

def col_order(df, colsList):
    # reorganizes the columns according to the colsList 
    cols = ([col for col in colsList if col in df] + [col for col in df if col not in colsList])
    return df[cols]

# Markdown
def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

# Streamlit page settings
st.set_page_config(page_title='CongraCNet 2', 
                layout="wide", 
                initial_sidebar_state='collapsed')

# Title
st.markdown(read_markdown_file("markdown/introduction.md"), unsafe_allow_html=True)

# Corpus selection
with st.beta_container():
    # Corpus select
    cS= st.selectbox('Corpus', s2n.corpusListDf['cS'], key='corpus')
    # Corpus data based on the selection
    try:
        corpSelect=s2n.corpusSelect(cS)
        corpus=corpSelect[1] #selected corpus
        language=corpSelect[2] # selected corpus language
        corpusID=corpSelect[3] # selected corpus ID
        gramRel=corpSelect[4] # selected corpus initial gramRel coordination type
        initial_lexeme= corpSelect[5] # selected corpus initial lexeme
        initial_pos= corpSelect[6] # selected corpus initial lexeme's pos
    except:
        pass
    ### Todo WIP - corpus info from api
    ### st.title('Corpus info')
    ### '', s2n.corpus_info(corpSelect[3], aS.userName, aS.apiKey)

    # Lemma and pos selection in 2 columns 
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


# Lemma pos requirements
assert lemma != '', st.error('Please choose a lemma')
assert pos != '-', st.error('Please choose a pos')

# Sidebar network parameters
st.sidebar.markdown(read_markdown_file("markdown/network_parameters.md"), unsafe_allow_html=True)

# Sidebar n-friend coocurences selection
limit = st.sidebar.number_input('Friend coocurences', 1, 100, 15,1) #initial limit

# Sidebar measure selection
measure=st.sidebar.selectbox('Measure',('score', 'freq')) #initial measure

# Source data harvest function
@st.cache(show_spinner=True)
def getInitData(lemma, pos, limit):
    'Harvests the data about the lexeme with a limit'
    try:
        s2n.corpusSketchHarvest(cS, [lemma], aS.userName, aS.apiKey, "wsketch?", pos, pos, corpus, limit, language, corpSelect[4])
        result= ("Sucess")
    except:
        result ='Some problems with harvesting, moving on...'
        pass
    return (result)

# Sidebar Source and Friend data harvest option
sourceHarvest = st.sidebar.checkbox('Source harvest', value=False, key='sourceHarvest') # Source harvest select
if sourceHarvest:
    if lemma and pos:
        st.spinner("Loading initial "+ str(limit)+ "coocurences and "+ measure+ 'as measure:'+ getInitData(lemma, pos, limit))

harvestCheck = st.sidebar.checkbox('Friend harvest', value= False) # Friend harvest select
if harvestCheck:
    harvestSelect = 'y'
else:
    harvestSelect = 'n'

# Frequency and Relative frequency of the source lemma in corpus
source_lema_data = cgcn_functions.source_lemma_freq(lemma, pos, corpusID, corpus, language, gramRel)
try:
    '***'+lempos+'***   **Frequency**: ', source_lema_data['freq'].values[0], '**Relative frequency (per million)**: ', source_lema_data['relFreq'].values[0]
except:
    pass



######################### Plotly Table instead of AG

def vis_table(df):
    headerColor = 'grey'
    rowEvenColor = 'lightgrey'
    rowOddColor = 'black'
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color='rgb(244,246,248)',
                    line_color='darkgray',
                    align='left'),
        cells=dict(values=[df[x] for x in df.columns],
                fill_color='white', line_color='darkgray',
                align='left'))])
    fig.update_layout(
    autosize=True,
    # width=500,
    height=20+(len(df)*70),
    margin=dict(
        l=10,
        r=10,
        b=10,
        t=10,
        pad=4
    ),
    paper_bgcolor="white",
)

    return fig

####### Wordnet integration
def print_val(word):
    'Takes a word and returns pos,neg,neu, valency values from Vader Lexicon'
    return json.dumps(wordnet_functions.get_valency_for_lempos([word])).replace('"','').replace('{', '').replace('}', '')

st.markdown("<h2 style='text-align:left;font-size:25px; padding-left:20px'><img src='http://emocnet.uniri.hr/wp-content/uploads/2021/02/037-pyramid_chart-1.png' alt='alt text' width=30>&nbsp;&nbsp;<b>Semantic dictionaries</b></h2>", unsafe_allow_html=True)
with st.beta_expander('WordNet synsets', expanded=True):
    # get synsets and hypernyms
    wn_hypernym=wordnet_functions.get_hypernyms([lempos], language)
    # get definitions
    wn_hypernym['source']= [(str(x)[8:-2]) for x in wn_hypernym['source']]
    wn_hypernym['target']= [(str(x)[8:-2]) for x in wn_hypernym['target']]
    wn_hypernym['s_definition']= [wordnet_functions.get_definition(x) for x in wn_hypernym['source']]
    wn_hypernym['t_definition']= [wordnet_functions.get_definition(x) for x in wn_hypernym['target']]
    wn_hypernym['valence_s']= [print_val(x) for x in wn_hypernym['source']]
    wn_hypernym['valence_t']= [print_val(x) for x in wn_hypernym['target']]

    st.markdown(wn_hypernym[['source','s_definition', 'target', 't_definition']].rename(columns={'source':'synset','s_definition':'definition', 'target':'hypernym', 't_definition':'definition'}).to_html(), unsafe_allow_html=True)
    
    # #header
    # cols_h = st.beta_columns((1,2,1,1,2,1))
    # cols_h[0].write('**Sense**')
    # cols_h[1].write('**Definition**')
    # cols_h[2].write('**Valency**')
    # cols_h[3].write('**Hypernym**')
    # cols_h[4].write('**Definition**')
    # cols_h[5].write('**Valency**')
    # #content   
    # for i in range(0, len(wn_hypernym)):
    #     cols = st.beta_columns((1,2,1,1,2,1))
    #     cols[0].markdown(f"<p style='font-size:1vw'><b>{str(wn_hypernym.source[i])[8:-2]}</b></p>", unsafe_allow_html=True)
    #     cols[1].markdown(f"<p style='font-size:1vw'><i>{wn_hypernym.s_definition[i]}</i></p>", unsafe_allow_html=True)
    #     cols[2].markdown(f"<p style='font-size:0.8vw'>{print_val(str(wn_hypernym.source[i])[8:-7])}</p>", unsafe_allow_html=True )
    #     cols[3].markdown(f"<p style='font-size:1vw'><b>{str(wn_hypernym.target[i])[8:-2]}</b></p>", unsafe_allow_html=True)
    #     cols[4].markdown(f"<p style='font-size:1vw'><i>{wn_hypernym.t_definition[i]}</i></p>", unsafe_allow_html=True)
    #     cols[5].markdown(f"<p style='font-size:0.8vw'>{print_val(str(wn_hypernym.target[i])[8:-7])}</p>", unsafe_allow_html=True)
            
with st.beta_expander('WordNet antonyms', expanded=True):
    antonym_df = wordnet_functions.get_antonyms_from_lempos([lempos], language)
    try:
        st.markdown(antonym_df.to_html(), unsafe_allow_html=True)
    except:
        pass

with st.beta_expander('WordNet domains'):
    unique_syns_words=wn_hypernym['source'].unique()# unique words
    for syn_word in unique_syns_words:
        syn_domains= ', '.join(spwf.get_domains_for_word(syn_word))
        cols= st.beta_columns((1,7.3))
        cols[0].markdown(f"<p style='font-size:1vw'><b>{syn_word}</b>", unsafe_allow_html=True) 
        cols[1].markdown(f"<p style='font-size:1vw'>{syn_domains}", unsafe_allow_html=True)



st.markdown("<h2 style='text-align:left;font-size:25px; padding-left:20px'><img src='https://kbi4.org/wp-content/uploads/2020/10/Sentiment-Analysis-using-MapReduce-Paradigm-from-Social-Media-Big-Data.jpg' alt='alt text' width=50>&nbsp;&nbsp;<b>Sentiment dictionaries</b></h2>", unsafe_allow_html=True)

st.sidebar.subheader('Sentiment calculation network')
limit_f= st.sidebar.number_input('Limit F value', 1, 50, limit,1)
limit_fof= st.sidebar.number_input('Limit FoF value', 1, limit, 5,1)
centrality_measures_list = ['sli', 'betweenness', 'degree','weighted_degree', 'pagerank']
sentiment_propagation_measure = st.sidebar.selectbox('Select centrality measure for ADV sentiment calcualtion', centrality_measures_list)
# update_measure = st.sidebar.selectbox('Select centrality measure for updating sentiment calcualtion', ['sli', 'betweenness', 'weighted_degree', ])
update_measure_com = st.sidebar.selectbox('Select centrality measure for updating GSV community sentiment calcualtion', centrality_measures_list)


with st.beta_expander('SenticNet 6', expanded=True):
    sentic6_orig= sentiment_functions.get_SenticNet_concept_df(lempos, language)
    'SenticNet 6 ODV: *Original Dictionary Value* for **', lempos,'**' 
    sentic6_orig= col_order(sentic6_orig, ['label', 'polarity_value', 'polarity_label'])
    st.markdown(sentic6_orig.to_html(), unsafe_allow_html=True)
    
    sentic6_val_df = sentiment_functions.calculate_sentic_values([lemma], pos, language, measure, corpus, corpusID, corpSelect[4], limit_f, limit_fof, harvestSelect)
    'SenticNet 6 ADV: *Assigned Dictionary Value* for **', lempos,'** based on the ', limit_f, ' friends and ', limit_fof, 'friends of friends graph of ', corpSelect[4], ' in ', corpusID,'.'
    sentic6_val_df= col_order(sentic6_val_df,['label', 'polarity_value'])
    st.dataframe(sentic6_val_df.style.background_gradient(cmap='plasma'))#.style.highlight_max(axis=0))

    if st.checkbox('Load the SentiNet6 difference dataset'):    
        d= pd.read_pickle('sentic_diff_sentic6_calcByBetweenness.pkl')
        'sentic_diff_sentic6_calcByBetweenness', d

# SentiWordsNet i sentiWords imaju samo engleski
if language=='en':
    with st.beta_expander('SentiWords 1.1.', expanded=False):
        try: 
            'SentiWords 1.1. ODV: *Original Dictionary Value* for **', lempos,'**' 
            originalSentiWords= sentiment_functions.get_sentiWords_values_df([lempos], sentiment_functions.sentiWords)
            st.markdown(originalSentiWords.to_html(), unsafe_allow_html=True)
            SentiWords11_val_df = sentiment_functions.calculate_sentiWords_values(sentiment_functions.sentiWords, [lemma], pos, language, measure, corpus, corpusID, corpSelect[4], limit_f, limit_fof, harvestSelect)
            'SentiWords 1.1. ADV: *Assigned Dictionary Value* for **', lempos,'** based on the ', limit_f, ' friends and ', limit_fof, 'friends of friends graph of ', corpSelect[4], ' in ', corpusID,'.'
            st.dataframe(SentiWords11_val_df.style.background_gradient(cmap='plasma'))#highlight_max(axis=0).highlight_min(axis=0))
        except:
            pass
    with st.beta_expander('SentiWordsNet', expanded=False):
        try:
            # import dataset sentiWordsNet
            'SentiWordsNet ODV: *Original Dictionary Value* for **', lempos,'**'
            sentiWordsNet=sentiment_functions.sentiWordsNet
            sentiWordsNet_value= sentiment_functions.get_word_sentiWordsNet_values_df(sentiWordsNet, [lempos])[['SynsetTerms', 'PosScore', 'NegScore', 'Gloss']]
            st.markdown(sentiWordsNet_value.to_html(), unsafe_allow_html=True)
        except:
            pass



# Network construction (lemma, pos, gramRel, measure, pos2)
st.markdown("<h2 style='text-align:left;font-size:25px; padding-left:20px'><img src='http://emocnet.uniri.hr/wp-content/uploads/2020/09/020-algorithm-300x300.png' alt='alt text' width=30>&nbsp;&nbsp;<b>Network construction</b></h2>", unsafe_allow_html=True)
# store list of grammatical relations for a source lemma (returns: gramRels, count)
listGramRelsS= cgcn_functions.lemmaGramRels(lemma, pos, corpusID, language)   
# GramRel list option
with st.beta_expander(str(len(listGramRelsS))+' grammatical relations for source lexeme: '+lemma+pos):
    # view list of grammatical relations   
    fig = px.bar(listGramRelsS.sort_values(by='count', ascending=True), 
                x="count", y="gramRels", orientation='h', opacity=0.7, log_x=False, height=600, 
                color='count', color_continuous_scale="blugrn",
                text='count')
    fig.update_layout(paper_bgcolor = 'rgba(0,0,0,0)', 
                    plot_bgcolor = 'rgba(240,240,240,0.5)',
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


# Network Visualization parameters
st.sidebar.markdown(read_markdown_file("markdown/visualization_parameters.md"), unsafe_allow_html=True)
# Vertex size selection 
vertexSizeType = st.sidebar.selectbox('Vertex size by', ('weighted_degree', 'degree', 'freq'))
if vertexSizeType == 'weighted_degree':
    if measure == 'score':
        vertexSizeValues= (0.0, 10.0, 5.0)
if vertexSizeType == 'degree':
    vertexSizeValues= (0.0, 10.0, 5.0)
vertexSize= st.sidebar.number_input('Vertex size', vertexSizeValues[0], vertexSizeValues[1], vertexSizeValues[2],0.5)
# Label size selection
vertexLabelSizeType = st.sidebar.selectbox('Label size by', ('weighted_degree', 'degree'))
if vertexLabelSizeType == 'weighted_degree':
    vertexLabelSizeValues= (1.0, 40.0, 10.0)
if vertexLabelSizeType == 'degree':
    vertexLabelSizeValues= (1.0, 40.0, 10.0)
vertexLabelSize= st.sidebar.number_input('Label size', vertexLabelSizeValues[0], vertexLabelSizeValues[1], vertexLabelSizeValues[2], 0.5)
# Edge size selection
edgeSize = st.sidebar.number_input('Edge size', 0.0,10.0,1.0, 0.5)
# Edge label size selection
edgeLabelSize = st.sidebar.number_input('Edge label size', 0.0,10.0,1.0,0.5)
# Layout selection
layout= st.sidebar.selectbox('Layout type', ( 'kk', 'fr', 'lgl', 'circle', 'drl', 'random', 'tree'))
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


# GramRel selection (selectbox, selection based on the listGramRelsS df, initial is set from the corpSelect[4])
gramRel1_col, gramRel2_col = st.beta_columns(2)
with gramRel1_col:
    gramRel= st.selectbox('Grammatical relation 1: ('+lemma+pos+')-[gr1]-(f)', listGramRelsS['gramRels'], list(listGramRelsS['gramRels']).index(corpSelect[4]))
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
    # mora izračunati sve FoF da bi izvukli pos2
    friends_list= gramReldf['friend'].str[0:-2].values
    def listGramRelsF(friends_list, pos2):
        df= pd.DataFrame()
        for x in friends_list[0:4]: # ograničeno je trenutno da izračuna samo prva 5, da ne troši resurse
            df= df.append(cgcn_functions.lemmaGramRels(x, pos2, corpusID, language))
        return df 
    # 'df', listGramRelsF(friends_list, pos2)['gramRels']
    listGramRelsF = listGramRelsF(friends_list, pos2)['gramRels'].unique()
    # 'listGramRelsF',listGramRelsF
    # 'indexed list', list(listGramRelsF).index(corpSelect[4]) 
    gramRel2= st.selectbox('Grammatical relation 2: ('+lemma+pos+')-[gr1]-(f)-[gr2]-(fof)', listGramRelsF, list(listGramRelsF).index(corpSelect[4]))
    
    # 'gramRel2', gramRel2#, gramRel2.values[0]
    try:
        gramRel2= gramRel2.values[0] #iz nekog razloga daje dvije vrijednosti u Series obliku
    except:
        gramRel2= gramRel2

    # GramReldf (function) selects a sample of the friend + gr2
    # '', language, friends_list[0], pos2, gramRel2, corpusID, measure
    # to find a proper pos3 we have to see what are typical pos for the gramrel2 for at least few friends of sources friend
    gramReldFof=pd.DataFrame()
    try:
        for friend in friends_list[0:3]:
            gramReldFof = gramReldFof.append(cgcn_functions.lemmaByGramRel(language, friend, pos2, gramRel2, corpusID, measure))
    except:
        for friend in friends_list[0:9]:
            gramReldFof = gramReldFof.append(cgcn_functions.lemmaByGramRel(language, friend, pos2, gramRel2, corpusID, measure))
    #'gramReldFof', gramReldFof
    # Filter pos3 (selectbox from gramReldf) 
    pos_fof = gramReldFof['friend'].str[-2:].unique()
        
    # # check if there are friends with multiple pos and choose 
    if len(pos_fof)>1:
        pos3='-'+st.selectbox('Filter the pos value of the friend lexeme', gramReldFof['friend'].str[-1].unique(), key='pos3_filter' )
    else:
        pos3=str(pos_fof[0])
    # 'gramRel2',gramRel2
    # 'pos', pos
    # 'pos2', pos2
    # 'pos3', pos3

# First degree network
# activate function friends_df() and store as friendCoocurences variable
friendCoocurences = cgcn_functions.friends_df(language, lemma, pos, corpus, corpusID, gramRel, pos2, limit, measure, harvestSelect, direction(gramRel,corpSelect[4]))
with st.beta_expander('1st degree: ('+lemma+pos+')-['+gramRel+']-(f), '+str(len(friendCoocurences))+' nodes ranked by: '+measure+', direction: '+direction(gramRel,corpSelect[4])):
    # 2 columns: graph, scatter    
    f_vis_col1, f_vis_col2 = st.beta_columns((1,1.8))
    with f_vis_col1:
        Fgraph = cgcn_functions.Fgraph(friendCoocurences, measure, direction(gramRel, corpSelect[4])) 
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
df_fof = cgcn_functions.FoFData(language, friendCoocurences, lemma, pos, corpus, corpusID, gramRel2, pos3, limit, measure, harvestSelect, direction(gramRel2, corpSelect[4]))
#store the result in a FoFgraph variable
FoFgraph=cgcn_functions.FoF_graph(df_fof, measure)  

with st.beta_expander('2nd degree: ('+lemma+pos+')-['+gramRel+']-(f)-['+gramRel2+']-(fof), '+ str(len(FoFgraph.vs))+' nodes ranked by: '+measure):
    # visualize network
    # if st.checkbox('FoF graph visualization'):
    st.plotly_chart(cgcn_functions.FgraphDraW_plotly(
        FoFgraph, layout, vertexSize, vertexLabelSize, 'Second degree network', edgeLabelSize),
        use_container_width=True, config=config)

    if st.checkbox('2nd degree dataset'):
        df_fof#[['source','friend', 'freq', 'score', 'gramRel']]



# Clustering 
st.markdown("<h2 style='text-align:left;font-size:25px; padding-left:20px'><img src='http://emocnet.uniri.hr/wp-content/uploads/2020/09/015-diagram-300x300.png' alt='alt text' width=30>&nbsp;&nbsp;<b>Clustering</b></h2>", unsafe_allow_html=True)
st.sidebar.markdown(read_markdown_file("markdown/clustering_parameters.md"), unsafe_allow_html=True)
algorithm = st.sidebar.selectbox('Cluster algorithm type', ('leiden', 'louvain'))
partitionType= st.sidebar.selectbox('Partition type', ('mvp', 'cpm'))
if partitionType == 'cpm':
    resolution = st.sidebar.number_input('Resolution', 0.0, 1.0, 0.5,0.02)
else:
    resolution = None

# Identify clusters in FoFgraph
clusterAlgo = cgcn_functions.clusterAlgo(FoFgraph, algorithm, partitionType, resolution)
with st.beta_expander('Clustered 2nd degree network with '+str(len(FoFgraph.vs))+' nodes and ' +str(len(FoFgraph.es))+' edges and '+\
                    str(len(clusterAlgo))+' clusters using algorithm = '+algorithm+\
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
    
    

    # Explore communities
    st.subheader('Community exploration')
    FoFgraph_df = pd.DataFrame({attr: FoFgraph.vs[attr] for attr in FoFgraph.vertex_attributes()})
    commsDataFrame=cgcn_functions.AlgoDataFrame(FoFgraph, clusterAlgo)
    algo_sort_by = st.multiselect('Sort nodes by', commsDataFrame.columns)
    try:    
        st.write(commsDataFrame.sort_values(by=algo_sort_by, ascending=False))
    except:
        pass


    
# Pruned version of FoFgraph
st.markdown("<h2 style='text-align:left;font-size:25px; padding-left:20px'><img src='http://emocnet.uniri.hr/wp-content/uploads/2020/09/007-funnel-300x300.png' alt='alt text' width=30>&nbsp;&nbsp;<b>Pruning</b></h2>", unsafe_allow_html=True)
# Sidebar Pruning
st.sidebar.markdown(read_markdown_file("markdown/pruning_parameters.md"), unsafe_allow_html=True)

# prune by betweenness
btwn = FoFgraph.betweenness()#weights='weight'
betw_ntile_prun=st.sidebar.number_input('Betweenness percentile', 0.0,100.0,50.0, 1.0)
ntile_betweenness = np.percentile(btwn, betw_ntile_prun)
pruned_vs = FoFgraph.vs.select([v for v, b in enumerate(btwn) if b >= ntile_betweenness])
pruned_graph = FoFgraph.subgraph(pruned_vs)

# prune by sli_importance
pruned_graph.vs['sli']= cgcn_functions.sli_importance(pruned_graph, igraph=True)
sli=pruned_graph.vs['sli']
sli_ntile_prun=st.sidebar.number_input('Sli importance percentile', 0.0,100.0,0.0,1.0)
ntile_sli = np.percentile(sli, sli_ntile_prun)
pruned_vs = FoFgraph.vs.select([v for v, b in enumerate(sli) if b >= ntile_sli])
pruned_graph = FoFgraph.subgraph(pruned_vs)

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
    cluster_resolution_prun=st.sidebar.slider('Resolution for pruned graph (0-low, 1-high)', 0.0,1.0,0.5,0.0)

clusterAlgo_prun= cgcn_functions.clusterAlgo(pruned_graph, algorithm, partitionTypePruned, cluster_resolution_prun)
with st.beta_expander('Pruned 2nd degree with '+str(len(pruned_graph.vs))+' nodes '+\
    str(len(pruned_graph.es))+' edges and '+\
    str(len(clusterAlgo))+' clusters '+\
    'using filters: degree >= '+str(prunedDegree[0])+\
        ', betweenness percentile: '+str(betw_ntile_prun)+\
        ', sli: '+str(sli_ntile_prun)+ ' clustering method: '+str(algorithm)+', partition type: '+str(partitionType)+', resolution: '+str(resolution), expanded=True):
    # prune by pruned_community
    pruned_DataFrame=cgcn_functions.AlgoDataFrame(pruned_graph, clusterAlgo_prun)
    out = (pruned_DataFrame.merge(commsDataFrame[['label', 'sli', 'pagerank']], left_on='label', right_on='label'))
    
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
    st.dataframe(pruned_DataFrame.reset_index(drop=True).style\
                                                        # .bar(subset=["betweenness","weighted_degree", "degree", 'pagerank'], color='rgba(201,201,201,0.5)')\
                                                        .background_gradient(cmap='plasma')\
                                                        # .highlight_max(axis=0, color='lightgray')
    
    )





######################## Antonyms
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

def scale_data(data, columns, min_max_scaler):
    for col in columns:
        data[col+'_norm'] = min_max_scaler.fit_transform(data[col].values.reshape(-1, 1))
    return data


with st.beta_expander('ConGraCNet + WordNet  antonyms', expanded=True):
    try:
        antonyms = antonym_df['ant_lempos'].unique().tolist()
        antonyms_samepos = [x for x in antonyms if x.endswith(pos)]
        # 'antonyms with same pos', antonyms_samepos
        # create a dataframe where each x in source+fof has antonym for each y in source-antonyms  
        antonym_DataFrame=pd.DataFrame()
        for ant in antonyms_samepos:
            df_ant = pruned_DataFrame[['label', 'pagerank', 'betweenness', 'degree', 'weighted_degree' , 'sli']]
            df_ant['antonym'] =  ant
            # df_ant = df_ant.append({'label': lempos, 'antonym': ant, 'pagerank': 1.0}, ignore_index= True)
            antonym_DataFrame= antonym_DataFrame.append(df_ant)
        antonym_DataFrame = antonym_DataFrame[['label','antonym', 'pagerank', 'betweenness', 'degree', 'weighted_degree', 'sli']]
        antonym_DataFrame_normalized = scale_data(antonym_DataFrame, ['pagerank', 'betweenness', 'degree', 'weighted_degree', 'sli'], min_max_scaler) 
        st.dataframe(antonym_DataFrame_normalized[['label', 'antonym', 'pagerank_norm', 'betweenness_norm', 'degree_norm', 'weighted_degree_norm', 'sli_norm', 'pagerank',  'betweenness', 'degree', 'weighted_degree']].reset_index(drop=True).style\
                                                # .bar(subset=["betweenness","weighted_degree", "degree", 'pagerank'], color='rgba(201,201,201,0.5)')\
                                                .background_gradient(cmap='plasma')\
                                                # .highlight_max(axis=0, color='lightgray')
            )
    
    except:
        pass










### Find categorical structure with is-a construction
st.markdown("<h2 style='text-align:left;font-size:25px; padding-left:20px'><img src='http://emocnet.uniri.hr/wp-content/uploads/2020/09/032-timeline-300x300.png' alt='alt text' width=30>&nbsp;&nbsp;<b>Community Labeling and Sentiment potential<b></h2>", unsafe_allow_html=True)
st.sidebar.markdown(read_markdown_file('markdown/labeling_parameters.md'), unsafe_allow_html=True)
def is_a_F(lemposSeries):
    #takes the lemma and finds: lemma is_a x_friend relation#
    df=pd.DataFrame()
    for row in lemposSeries:
        try:
            # df_friend= cgcn_functions.friends_df(language, row[0:-2], row[-2:], corpus, corpusID, '... is a "%w"', '-n', 25, measure, harvestSelect, 'directed')
            df_friend= cgcn_functions.friends_df(language, row[0:-2], row[-2:], corpus, corpusID, '"%w" is a ...', '-n', 25, measure, harvestSelect, 'directed')
            df= df.append(df_friend)
        except:
            pass        
    return df


# Get set communities
comms = pruned_DataFrame['commPart'].unique()
counter=-1
# Collect sentic values for all communities
df_all_sentic_communities=pd.DataFrame()

#select how to sort the community lexemes in is_a graphs 
sort_is_a= st.sidebar.multiselect('Sort is_a dataset by ', ['degree',  'pagerank', 'weighted_degree', 'betweenness', 'label'], default= [ 'degree', 'pagerank', 'weighted_degree', 'betweenness', 'label']) 
# 'eigen_c', 'sbb_importance', 'sbb2_importance',
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
label_num=st.sidebar.number_input('Choose number of is_a labels', 1,10,2,1, key='labels_number')


# display community members
ch_is_a_comm_members=st.sidebar.checkbox('Display community members', value=True)
ch_is_a_network_df=st.sidebar.checkbox('Display first degree is_a dataframe', value=True)
ch_is_a_network_graph=st.sidebar.checkbox('Display first degree is_a network', value=True)
ch_is_a_network_second=st.sidebar.checkbox('Display second degree is_a dataframe', value=False)
ch_is_a_wordnet_hyper=st.sidebar.checkbox('Display wordnet hypernym results', value=True)
ch_is_a_wordnet_domains=st.sidebar.checkbox('Display wodnet domains for the community', value=True)

for comm in comms:
    counter=counter+1
    # members of a class 
    items= pruned_DataFrame[pruned_DataFrame['commPart']==comm]['label']
    st.markdown('<h3 style="background-color:#F0F2F6; padding-left:5px">Community '+str(counter+1)+':</h3> <p style="background-color:#F4F6F8; padding-left:5px"><i>'+', '.join(items.values)+'</i></p>', unsafe_allow_html=True)
    
    # for items in a community get first degree collocates

    try: # makni try da bi se opet vidjele korpusne
        # create Friend graph from is_a_net dataset 
        is_a_net= is_a_F(items) # dataset with ['source', 'friend', measure]
        is_a_net_network = cgcn_functions.Fgraph(is_a_net, measure, 'directed')
        # is_a_net_network['sli']= cgcn_functions.sli_importance(is_a_net_network)
        is_a_net_df = pd.DataFrame({attr: is_a_net_network.vs[attr] for attr in is_a_net_network.vertex_attributes()}).sort_values(by = sort_is_a, ascending=False).drop(['color'], axis=1)
        
        with st.beta_expander('> Corpus-based labeling of community '+str(counter+1)+': '+lempos+" is related to "+", ".join(is_a_net_df.head()['label'][0:label_num]).upper()):
            try:
                # st.markdown("<h3 style='background-color:aliceblue'><i>&nbsp;&nbsp;"+lempos+"</i> is related to "+", ".join(is_a_net_df.head()['label'][0:label_num]).upper()+"</h3>", unsafe_allow_html=True)    
                if ch_is_a_comm_members:
                    st.markdown('<p>'+str(len(items))+' associated members of a <b>community '+str(counter+1)+"</b>: <i>"+', '.join(items.values)+'</i></p>', unsafe_allow_html=True) 
                col_isa1, col_isa2= st.beta_columns([1,1])
                with col_isa1:
                    if ch_is_a_network_df:
                        st.markdown('**Community '+str(counter+1)+'** *is_a ... label* of first degree network (best 20 ranked):')
                        st.dataframe(is_a_net_df[['label', 'degree', 'weighted_degree', 'pagerank','betweenness']][0:20].style.background_gradient(cmap='plasma'))                     
                with col_isa2:
                    if ch_is_a_network_graph:
                        st.markdown('<p>First degree network of <b>community '+str(counter+1)+'</b> <i>is_a ... label</i>:</p>', unsafe_allow_html=True)
                        is_a_net_network_Fgraph=cgcn_functions.FgraphDraW_plotly(is_a_net_network, layout, vertexSize, vertexLabelSize, 'Is_a network for community '+str(counter+1), edgeLabelSize)
                        st.plotly_chart(is_a_net_network_Fgraph, use_container_width=True)
                if ch_is_a_network_second:
                    # get second degree is_a collocations
                    is_a_net2 =  is_a_F(is_a_net_df['label'])
                    # create second degree is_a graph
                    is_a_net2_network= cgcn_functions.Fgraph(is_a_net2, measure, 'directed')
                    # create dataframe for second degree
                    is_a_net2_df= cgcn_functions.df_from_graph(is_a_net2_network)
                    st.markdown('<p>Second degree nodes of <b>community '+str(counter+1)+'</b> <i>is_a ...label is_a...label</i> network:</p>', unsafe_allow_html=True)
                    st.dataframe(is_a_net2_df.sort_values(by=sort_is_a, ascending=False))
                    
            except:
                pass
    except:
        pass

    try: 
        # Wordnet label calculation
        if ch_is_a_wordnet_hyper:
            wn_hypernyms= wordnet_functions.get_hypernyms(items, language)#returns df with source, target, weight#(is_a_net_df['label'])
            wn_hypernyms['weight']=1
            wn_hypernyms['source']=wn_hypernyms['source'].astype(str).str[8:-2]
            wn_hypernyms['target']=wn_hypernyms['target'].astype(str).str[8:-2]
            #Create a nx graph from wn_hypernyms
            tuples = [tuple(x) for x in wn_hypernyms[['source', 'target', 'weight']].values]
            G=ig.Graph.TupleList(tuples, directed = True, edge_attrs=['weight'], vertex_name_attr='name', weights=False)
            #create vertex labels from name attr
            G.vs["label"]= G.vs["name"]
            G.vs["degree"]=G.vs.degree()
            G.vs["pagerank"]=G.vs.pagerank(directed=False, weights='weight')
            G.vs["personalized_pagerank"]= G.vs.personalized_pagerank(directed=False, weights='weight')
            G.vs["betweenness"] = G.betweenness()
            # G.vs['eigen_c'] = G.eigenvector_centrality(directed=True, scale=True, weights= 'weight', return_eigenvalue=False)
            G.vs["weighted_degree"] = G.strength(G.vs["label"], weights='weight', mode='ALL')     
            wndfg= pd.DataFrame()
            wndfg["label"]=G.vs["label"]
            wndfg["syn"]= [str(x) for x in G.vs["label"]]
            wndfg["definition"]=[wordnet_functions.get_definition(x) for x in wndfg["syn"]]
            # wndfg['eigen_c']=G.vs['eigen_c']
            wndfg["degree"]=G.vs["degree"]
            wndfg["pagerank"]=G.vs["pagerank"]

            # add wordnet sentiment to a hypernym graph
            # wndfg['sentiwordnet_pos'] = sentiment_functions.get_word_sentiWordsNet_values_df(sentiWordsNet, x.split)
           
            wndfg= wndfg.sort_values(by=['degree', 'pagerank'], ascending=False) #'eigen_c'
            with st.beta_expander('> WordNet hypernym-based labeling of community '+str(counter+1)+': '+lempos+" is related to synsets "+", ".join(wndfg['syn'][0:label_num]).upper()):
                # st.markdown("<p style='background-color:aliceblue; padding-left:5px'>Wordnet <i>hypernym synsets</i> for <b>class "+str(counter)+"</b>: "+str(', '.join(wndfg['syn'].astype(str).str.slice(start=0,stop=-5)[0:int(label_num)]).upper())+"</p>", unsafe_allow_html=True) 
                '**First degree hypernym network**: Lemma --> Hypernym(Synset) (20 best ranked)'
                st.dataframe(wndfg[0:19].style.background_gradient(cmap='plasma', subset=['degree', 'pagerank']))          
                
                if st.checkbox('Wordnet hypernyms for words in class '+str(counter)):
                    st.dataframe(wn_hypernyms) #.groupby('target').count()['source'].sort_values(ascending=False)
                    'Hypernym graph', G
                if ch_is_a_wordnet_domains:
                    st.subheader('WordNet domains')
                    tuples_d=[]
                    domainsList=[]
                    G = ig.Graph()
                    # Get synwords from items using gethypernyms 
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
                    'Wordnet *domains* for **community '+str(counter+1)+'**:'
                    st.dataframe(domSeries_count)

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
    except:
        pass

    

    with st.beta_expander('> Sentiment calculations for community '+str(counter+1)):
        update_measure=  sentiment_propagation_measure # bilo je 'betweenness' služi za izračunavanje udjela snage čvorova neke zajednice u cijelom grafu
        node_propagation_measure= sentiment_propagation_measure
            
        #########Vader
        st.markdown('**Vader joint** sentiment for community '+str(counter+1))
        valency = wordnet_functions.get_valency_for_lempos(wndfg["syn"])
        st.markdown(json.dumps(valency).replace('"','').replace('{', '').replace('}', ''))       
        
        col_vad1, col_vad2 = st.beta_columns([2,2])
        with col_vad1:
            st.markdown('<b>Vader ODV</b>: Original sentiment values for community '+str(counter+1), unsafe_allow_html=True) 
            vaderODV= wordnet_functions.get_wnValency_for_lempos_df(items, language)
            vaderODV = col_order(vaderODV, ['label'])
            st.dataframe(vaderODV)
            # st.markdown(vaderODV.to_html(), unsafe_allow_html=True)
        with col_vad2:
            vaderODVSrednjica= sentiment_functions.get_vader_srednjica_df(vaderODV,  pruned_DataFrame[pruned_DataFrame['label'].isin(items)]['weighted_degree'], 'weighted_degree')
            st.markdown('**Vader GSV**: Graph sentiment value for subgraph for community '+str(counter+1), unsafe_allow_html=True) 
            st.dataframe(vaderODVSrednjica)
        
        st.markdown('---')

        #######sentiWords_1.1
        col_sw1, col_sw2 = st.beta_columns([2,2])
        with col_sw1:
            st.markdown('<b>SentiWords 1.1 ODV</b>: Original sentiment values for community '+str(counter+1), unsafe_allow_html=True)
            sentiWordsDict = sentiment_functions.sentiWords
            sentiWordsDict_items= sentiment_functions.get_sentiWords_values_df(items, sentiWordsDict) 
            st.dataframe(sentiWordsDict_items)
        with col_sw2:
            # srednjica  
            sentiWordsSrednjica= sentiment_functions.get_sentiWords_srednjica_df(sentiWordsDict_items,  pruned_DataFrame[pruned_DataFrame['label'].isin(items)]['weighted_degree'], 'weighted_degree')
            st.markdown('**SentiWords 1.1 GSV**: Graph sentiment value for subgraph for community '+str(counter+1))         
            st.dataframe(sentiWordsSrednjica)
        
        st.markdown('---')

        ############# Sentic Addon
        sentic_concepts_df= sentiment_functions.get_SenticNet_c_df(items, language).reset_index(drop=True)
        if not 'polarity_value' in sentic_concepts_df.columns:
            sentic_concepts_df['polarity_value'] = ''
        
        # if every concept has sentic value 
        if not sentic_concepts_df['polarity_value'].isna().any():
            col_sntc1, col_sntc2 = st.beta_columns([2,2])
            with col_sntc1:
                st.markdown('**SenticNet 6 ODV**: Original dictionary values for community '+str(counter+1))
                sentic_concepts_df= col_order(sentic_concepts_df,['label', 'polarity_value', 'polarity_label'])
                st.dataframe(sentic_concepts_df)
            with col_sntc2:
                '**SenticNet 6 GSV**: Graph sentiment values based on the *original* (ODV) for the community '+str(counter+1) 
                #append to df_all_sentic_communities for representing sentiment potential 1 (ima još jedna dolje isti odjeljak koda za drugu opciju kad nema svih čvorova)                
                sentic_value = sentiment_functions.get_graph_sentic_values(sentic_concepts_df,pruned_DataFrame)        
                sentic_value['comm'] = counter+1
                sentic_value['centrality_measure']=pruned_DataFrame[pruned_DataFrame['commPart']==counter][sentic_value.index].sum()
                sentic_value['WordNet_hypernym_label']= ', '.join(wndfg["syn"][0:2].values.tolist())
                st.dataframe(sentic_value)
                df_all_sentic_communities = df_all_sentic_communities.append(sentic_value)
            
        # if any concept has NOT sentic value
        if sentic_concepts_df['polarity_value'].isna().any():
            col_sntc1, col_sntc2 = st.beta_columns([2,2])
            with col_sntc1:

                # '**SenticNet 6 GSV**: Graph sentiment values based on the *original* (ODV) for the community '+str(counter+1)             
                lemmas_not_included= sentic_concepts_df[sentic_concepts_df['polarity_value'].isna()]['label']
                ######### Procesiraj sentic non processed čvorove i napravi sentic vrijednosti pomoću concept(FoFgraf)                 
                sentic_concepts_updated_df= pd.DataFrame()
                for concept in sentic_concepts_df[sentic_concepts_df['polarity_value'].isna()]['label']:
                    try:
                        # 1 get FoF data n= 15
                        limit_snp_f=15 # limit za first degree
                        limit_snp_fof=5 # limit za second degree
                        # f_snp_friends
                        f_snp_df = cgcn_functions.friends_df(language, concept[0:-2], concept[-2:], corpus, corpusID, corpSelect[4], concept[-2:], limit_snp_f, measure, harvestSelect, 'undirected')
                        # fof_snp_friends
                        fof_snp_df= cgcn_functions.FoFData(language, f_snp_df, concept[0:-2], concept[-2:], corpus, corpusID, corpSelect[4], concept[-2:], limit_snp_fof, measure, harvestSelect, 'undirected')
                        fof_snp_graph = cgcn_functions.FoF_graph(fof_snp_df, measure)
                        fof_snp_graph_df= cgcn_functions.df_from_graph(fof_snp_graph)
                        sentic_snp = sentiment_functions.get_SenticNet_c_df(fof_snp_graph.vs["label"], language)
                        # 'sentic_snp '+concept, sentic_snp
                        # sentic vrijednosti se računaju na temelju sentic not processed čvorova i njihovih fof_snp_grafova
                        sentic_snp_value=sentiment_functions.get_graph_sentic_values(sentic_snp, fof_snp_graph_df)
                        sentic_snp_value['label']= concept
                        # 'sentic_snp value for '+concept, sentic_snp_value#.loc['degree']
                        sentic_concepts_updated_df=sentic_concepts_updated_df.append(sentic_snp_value) 
                    except:
                        pass
            
                # 'sentic_concepts_updated_df',sentic_concepts_updated_df.loc['degree'].reset_index(drop=True)
                # updated sentic values for all nodes in a graph 
                sentic_concepts_all_df= sentic_concepts_df[sentic_concepts_df['polarity_value'].notna()].append(sentic_concepts_updated_df.loc[node_propagation_measure]).reset_index(drop=True)#[['label', 'polarity_value', 'introspection', 'temper', 'attitude', 'sensitivity', 'polarity_label', 'moodtags', 'semantics']]        
                sentic_concepts_all_df = col_order(sentic_concepts_all_df,['label', 'polarity_value', 'polarity_label'])
                
                
                st.markdown('**SenticNet 6 ODV+ADV**: Original + Assigned dictionary values based on '+ node_propagation_measure + ' FoF for nodes in the community '+str(counter+1))
                st.dataframe(sentic_concepts_all_df)
                st.text('Assigned sentic values for '+str(len(lemmas_not_included)) +' lemmas not included in sentic dictionary based on the '+ node_propagation_measure +' measure.')
            
                # if st.checkbox('Represent statistics', key='polarity_statistics'):
                #     col_pol1, col_pol2 = st.beta_columns([1,3])
                #     with col_pol1:
                #         'Polarity_value statistics', (sentic_concepts_all_df['polarity_value'].astype(float).describe().reset_index(), height=300, key='sentic_all'+str(counter))
                #     with col_pol2:
                #         dbox = sentic_concepts_all_df['polarity_value'].astype(float).describe()
                #         trace=go.Box(
                #             y = dbox[1:],
                #             name = 'Polarity values'
                #         )
                #         data=trace
                #         layout=go.Layout()
                #         fig= go.Figure(data=data, layout=layout)
                #         st.plotly_chart(fig)
                
            with col_sntc2:
                sc_all_values= sentiment_functions.get_graph_sentic_values(sentic_concepts_all_df,pruned_DataFrame) 
                #append to df_all_sentic_communities for representing sentiment potential
                sc_all_values['comm']= counter+1
                # suma vrijednosti centralnosti po zajednici po mjeri
                sc_all_values['centrality_measure']= [x for x in pruned_DataFrame[pruned_DataFrame['commPart']==counter][sc_all_values.index].sum()]
                sc_all_values['WordNet_hypernym_label']= ', '.join(wndfg["syn"][0:2].values.tolist())
                ## for all values, calculate srednjica for all measures
                gvs_com_df = sentiment_functions.get_sentic_srednjica_comm(sentic_concepts_all_df, pruned_DataFrame[pruned_DataFrame['label'].isin(sentic_concepts_all_df['label'])], ['sli', 'pagerank', 'degree', 'weighted_degree', 'betweenness'])
                gvs_com_df= gvs_com_df.combine_first(sc_all_values) # umetne sentic izračune sa svim vrijednostima iz gsv_com_df na sc_all_values
                try: # za engleski set neka proba urediti kolumne da budu iste
                    gvs_com_df= col_order(gvs_com_df,['label', 'polarity_value', 'introspection', 'temper'])
                except:
                    pass
                
                '**SenticNet 6 GSV**: Graph sentiment values based on the *original* (ODV) + *assigned* (ADV) dictionary values ' + ' for the community '+str(counter+1)
                st.dataframe(gvs_com_df)
                df_all_sentic_communities=df_all_sentic_communities.append(gvs_com_df) # update all values from calculated



st.markdown('<h3 style="background-color:#F0F2F6; padding-left:5px">SenticNet 6 SP: Sentiment potential (SP) for <i>'+lempos+'</i> in <i>'+corpusID+'</i></h3><p>Polarity_value calculated with '+ update_measure_com + ' Sentic values for the community.</p>', unsafe_allow_html=True)
# 'Polarity_value calculated with '+ update_measure_com + ' Sentic values for the community.'
df= df_all_sentic_communities.loc[update_measure_com].reset_index()

# napravi vrijednosti postotka
df['postotak']=(df['centrality_measure']/df['centrality_measure'].sum()) #baca grešku ako se izolira jedna community
# napravi vrijednosti srednjice zajednica gsv_comm
df['gsv_comm']= cgcn_functions.srednjica(df['polarity_value'], df['centrality_measure'] )
# dodaj izvornu vrijednost sentica
try:
    df['odv'] = sentic6_orig['polarity_value'][0]
except:
    pass

try:
    df['adv']= sentic6_val_df.loc[node_propagation_measure]['polarity_value'] ##assigned đđđ
except:
    pass

fig_sentiment_potential_scatter = px.scatter(df, 
                x=df['comm'], y='polarity_value',
                range_y=[-1,1],
                size='centrality_measure',
                size_max=60, 
                # width=,
                hover_data=['WordNet_hypernym_label', 'polarity_value', 'centrality_measure', 'postotak'], 
                color='polarity_value',
                range_color=[-1,1],
                color_continuous_scale=[(0, "black"), (0.33, "blue"), (0.66, "red"), (1, "yellow")],
                # text=[f"({round(row['postotak']*100,1)}%)" for index, row in df.iterrows()],
                labels={'WordNet_hypernym_label':'WordNet_hypernym', 'postotak':'Centrality peracentage', 'comm':'Community'},           
                height=500, #width=800,
                opacity=0.7, trendline_color_override='rgba(255,0,0,0.1)'
                # trendline ='ols'
                )
# add a GSV line green with annotation
try:
    fig_sentiment_potential_scatter.add_trace(go.Scatter(x=df['comm'], y=df['gsv_comm'], mode='lines', text='GSV', opacity=0.4))
    fig_sentiment_potential_scatter.add_annotation(text='ASP:'+str(round(df['gsv_comm'][0],2)), 
                                                yshift=0,x=1.3, y=df['gsv_comm'][0],showarrow=True,
                                                font=dict(color='gray',size=10))
    fig_sentiment_potential_scatter.update_layout(showlegend=False)
except:
    pass
# add a Sentic value with annotation
try:
    fig_sentiment_potential_scatter.add_trace(go.Scatter(x=df['comm'], y=df['odv'], mode='lines', text='Sentic', opacity=0.4))
    fig_sentiment_potential_scatter.add_annotation(text='ODV:'+str(df['odv'][0]), yshift= 0,x=1.7, y=df['odv'][0],showarrow=True,
                                                font=dict(color='gray',size=10))
# fig_sentiment_potential_scatter.update_layout(showlegend=False)
except:
    pass
# ADV value add
try:
    fig_sentiment_potential_scatter.add_trace(go.Scatter(x=df['comm'], y=df['adv'], mode='lines', text='Sentic', opacity=0.4))
    fig_sentiment_potential_scatter.add_annotation(text='ADV:'+str(round(df['adv'][0],2)), yshift= 0,x=1.5, y=df['adv'][0],showarrow=True,
                                                font=dict(color='gray',size=10))
# fig_sentiment_potential_scatter.update_layout(showlegend=False)
except:
    pass

#dodaj anotacije
for index, row in df.iterrows():
    # prvi hypernym
    fig_sentiment_potential_scatter.add_annotation(
                    text=row["WordNet_hypernym_label"].split(',')[0],
                    yshift=70,x=row['comm'], y=row['polarity_value'],showarrow=False)
    try:
        fig_sentiment_potential_scatter.add_annotation(
                    text=row["WordNet_hypernym_label"].split(',')[1],
                    yshift=55,x=row['comm'], y=row['polarity_value'],showarrow=False)
    except:
        pass
    fig_sentiment_potential_scatter.add_annotation(
                    text=str(round(row["postotak"]*100,1))+'%',
                    yshift=-50,x=row['comm'], y=row['polarity_value'],showarrow=False)
    fig_sentiment_potential_scatter.add_annotation(
                    text=round(row["polarity_value"],2),
                    font=dict(color="white",size=12),
                    yshift=-1,x=row['comm'], y=row['polarity_value'],showarrow=False)

fig_sentiment_potential_scatter.update_traces(
                                marker=dict(line=dict(width=2,color='DarkSlateGrey')),
                                selector=dict(mode='markers'))
fig_sentiment_potential_scatter.update_layout(paper_bgcolor = 'white', plot_bgcolor = 'rgba(226,226,226,0.5)',
                                        xaxis=dict(showline=True, zeroline=True, 
                                        showgrid=True, showticklabels=True, 
                                        tickvals = df['comm'],
                                        title=''
                                        # 'Polarity value ' +str(len(df['comm']))+ ' for '+lempos+' in '+ corpusID\
                                        #     +'. Calculated with limit_f: '+str(limit_f)+', limit_fof: '+str(limit_fof) + ', update measure for community: '+ update_measure_com +' (%)'
                                            ),
                                        
                                        )
config={"displayModeBar": False, "showTips": True}
st.plotly_chart(fig_sentiment_potential_scatter, use_container_width=True, config=config)


final_sentic = df_all_sentic_communities.loc[update_measure_com]
# 'Sentic sentiment values for all communities of **', lempos,'** based on the', update_measure_com, ' centrality measures'
# izračunaj završnu vrijednost 
final_sentic_srednjica= cgcn_functions.srednjica(final_sentic['polarity_value'], final_sentic['centrality_measure'] )
final_sentic['ASP_lempos']=final_sentic_srednjica
final_sentic['comm_difference']=final_sentic['polarity_value']-final_sentic['ASP_lempos']


# 'Final sentiment value for ', lempos, 'in', corpusID, ' abstracted on all communities for ', update_measure_com
# st.markdown('**'+str(round(final_sentic_srednjica, 4))+'**')


# text
# st.subheader('*Sentiment potential* (SP) for **'+lempos+'** in '+corpusID)
st.markdown('<h4 style="background-color:white; padding-left:0px">Sentiment potential (SP) for <i>'+lempos+'</i> in <i>'+corpusID+'</i></h4>', unsafe_allow_html=True)
final1_column, final2_column = st.beta_columns((1,3))
with final1_column: 
   
    '*Assigned sentiment potential* (ASP) ', round(final_sentic_srednjica, 4)
    try:
        '*Original sentiment value* (ODV) ', float(sentic6_orig['polarity_value'][0])
    except:
        '*Original sentiment value* (ODV) **not present**.'
        pass
    '*Assigned sentiment value* (ADV) ', round(sentic6_val_df.loc[node_propagation_measure]['polarity_value'],4)
with final2_column:
    for index, row in final_sentic.iterrows():
        '*Community*', row['comm'],' labeled: **'+row['WordNet_hypernym_label']+'**. *Graph sentiment value (GSV)*', round(row['polarity_value'],4), '*Percentage of the seed lexical graph*', round(row['centrality_measure']/final_sentic['centrality_measure'].sum()*100, 4)
'Polarity value ADV calculated with*', node_propagation_measure+'*. GSV, ASV calculated with*',  update_measure_com+'*.'

if st.checkbox('Represent dataset with SP values', key='SP_final_dataset'):
    # df_all_sentic_communities['centrality_measure']= pruned_DataFrame[sentiment_comm_measure].sum()
    'Sentic sentiment values of **', lempos,'** for all sentic communities accros all centrality measures'
    df_all_sentic_communities
    # AgGrid(df_all_sentic_communities.reset_index(), height=170, key='da_all_sentics') #cellStyle="--x_x--0_0-- function(params) { if (params.value == 'polarity_value') { return { 'color': 'white', 'backgroundColor': 'darkred' } } else { return { 'color': 'black', 'backgroundColor': 'white' } } }; --x_x--0_0--")
    # final value of the seed graph node
    final_sentic[['ASP_lempos', 'polarity_value', 'centrality_measure', 'comm', 'comm_difference', 'WordNet_hypernym_label']]
    # AgGrid(final_sentic[['ASP_lempos', 'polarity_value', 'centrality_measure', 'comm', 'comm_difference', 'WordNet_hypernym_label']], height=170, key='finalsentiment')

            # st.text('Dictionary approach')
            # sentic_list= sentiment_functions.get_SenticNet_c_list(items, language)
            # 'sentic_list - rječnik riječi koje su procesirane', pd.DataFrame(sentic_list)
            # 'sentic to dict', sentiment_functions.get_SenticNet_c_df(items, language)
            # # sentic dataframe values for existing concepts with node measures 
            # sentic_values_df = pd.DataFrame()
            # for mnm in ['pagerank', 'degree', 'weighted_degree']: #'sbb_importance', 'sbb2_importance'
            #     mod_node_measure= mnm
            #     node_importance_values_list = pruned_DataFrame[pruned_DataFrame["label"].isin([str(x[0]) for x in sentic_list])][mod_node_measure].astype('float').tolist()
            #     sentic_values_df= sentic_values_df.append(sentiment_functions.make_sentic_df(sentic_list, node_importance_values_list, mnm))
            # 'sentic_values_df values for existing concepts', sentic_values_df 
            # 'sentic polarity_value list', pd.DataFrame(sentic_list)[1]
            
            # 'describe sentic polarity_value', pd.Series(sentic_list[1])#.astype(float).describe()
            
            # # riječi koje nisu procesirane
            # 'sentic_list - riječi koje su procesirane #dodaj i vrijednosti u taj table', pd.DataFrame(sentic_list)
            # sentic_non_processed= items[items.isin(pd.DataFrame(sentic_list)[0])== False].dropna()
            # 'items koji nisu procesirani', sentic_non_processed
            
            # ######### Procesiraj čvorove koji nemaju sentic vrijednosti iz njegovog FoF grafa 
            # if sentic_non_processed.any():
                
            #     for concept in sentic_non_processed:
            #         # 1 get FoF data n= 15
            #         # 'sentic_non_processed 0', concept
            #         limit_snp_f=15 # limit za first degree
            #         limit_snp_fof=5 # limit za second degree
            #         f_snp_df = cgcn_functions.friends_df(language, concept.split('-')[0], '-'+concept.split('-')[1], corpus, corpusID, corpSelect[4], concept.split('-')[1], limit_snp_f, measure, harvestSelect, 'undirected')
            #         # 'f_snp_friends', f_snp_df
            #         fof_snp_df= cgcn_functions.FoFData(language, f_snp_df, concept.split('-')[0], '-'+concept.split('-')[1], corpus, corpusID, corpSelect[4], concept.split('-')[1], limit_snp_fof, measure, harvestSelect, 'undirected')
            #         # 'fof_snp_friends', fof_snp_df
            #         fof_snp_graph = cgcn_functions.FoF_graph(fof_snp_df, measure)
            #         fof_snp_graph_df= cgcn_functions.df_from_graph(fof_snp_graph)
            #         # 'fof_snp_graph_df',fof_snp_graph_df
            #         sentic_snp= sentiment_functions.get_SenticNet_c_list(fof_snp_graph.vs["label"], language)
            #         sentic_non_processed_df=pd.DataFrame()
            #         for mnm in ['pagerank', 'degree', 'weighted_degree']: #'sbb_importance', 'sbb2_importance'
            #             mod_node_measure= mnm
            #             node_importance_np_values_list = fof_snp_graph_df[fof_snp_graph_df["label"].isin([str(x[0]) for x in sentic_snp])][mod_node_measure].astype('float').tolist()
            #             sentic_non_processed_df= sentic_non_processed_df.append(sentiment_functions.make_sentic_df(sentic_snp, node_importance_np_values_list, mnm))
            #         'sentic_non_processed_df for '+concept, sentic_non_processed_df

                    


            #         # 'sentic_snp list for missing '+concept, sentic_snp
            #         # 'snp_by_degree for '+concept, sentiment_functions.make_sentic_df(sentic_snp, fof_snp_graph.vs["degree"], 'degree')
                    

            #         #sentic_value_for_missing_snp= 
            #         # valency_snp = wordnet_functions.get_valency_for_lempos(fof_snp_graph.vs["label"])
            #         # 'valency_snp',valency_snp
            #         # fof_snp_graph = cgcn_functions.FoF_graph()
            #         # 2 dobijemo vrijednosti u FoF grafu get_SenticNet_c_list(concept_list, language)

            #         # 3 izračunamo srednju vrijednost sentic vrijednosti pomoću srednjica srednjica(val_list, node_importance_values_list)
            #         # 4 odrediti po kojoj node importance vrijednosti uzeti sentic vrijednosti
            #         # gdje bilježim te vrijednosti?

            # val_list= [float(x[1]['polarity_value']) for x in sentic_list]
            # # 'Sentic community **polarity value** modified by '+ mod_node_measure, cgcn_functions.srednjica(val_list, node_importance_values_list )
                    
            

            # # # sentic values for all
            # # sentic_all_df = pd.DataFrame()
            # # for mnm in ['pagerank', 'degree', 'weighted_degree']: #'sbb_importance', 'sbb2_importance'
            # #     mod_node_measure= mnm
            # #     node_importance_all_values_list = node_importance_np_values_list+node_importance_values_list
            # #     sentic_all_df= sentic_df.append(sentiment_functions.make_sentic_df(sentic_snp+sentic_list, node_importance_values_list, mnm))
            
            # # 'sentic_df for existing concepts', sentic__all_df 
             



            
       # st.markdown('----')
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
# napraviti da se samo koordinacija izračunava kao neusmjereni graf a svi ostali kao usmjeren