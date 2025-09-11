#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 12:48:45 2019
@author: Benedikt Perak
"""
# I Database Connect section

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

# =============================================================================
# Function for extracting data frame of coocurence lemposFs, score, freq
# =============================================================================

import pandas as pd
import numpy as np
import plotly
from plotly.offline import plot
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.io as pio

def and_or_coocurences(lemma, pos, corpus, limit):
    if corpus == "hrwac":
        q='''
        MATCH p=(lempos:Lemma{lempos:$lemma+$pos, language:"hr"})-[r:`koordinacija`]-(lemposF:Lemma) 
        WHERE lemposF.lempos ENDS WITH $pos 
        WITH lemposF.lempos AS lemposF, r.score_hrwac22 as score, r.count_hrwac22 as freq 
        ORDER BY score DESC LIMIT $limit
        RETURN distinct(lemposF) AS lemposFs,  freq, score'''
    
    if corpus == "ententen13":
        q='''
        MATCH p=(lempos:Lemma{lempos:$lemma+$pos, language:"en"})-[r:`"%w" and/or ...`]-(lemposF:Lemma) 
        WHERE lemposF.lempos ENDS WITH $pos 
        WITH lemposF.lempos AS lemposF, r.score_ententen13_tt2_1 as score, r.count_ententen13_tt2_1 as freq 
        ORDER BY score DESC LIMIT $limit
        RETURN distinct(lemposF) AS lemposFs,  freq, score
        '''
    if corpus == "europarl7":
        q='''
        MATCH p=(lempos:Lemma{lempos:$lemma+$pos, language:"en"})-[r:`and/or`]-(lemposF:Lemma) 
        WHERE lemposF.lempos ENDS WITH $pos 
        WITH lemposF.lempos AS lemposF, r.score_europarl7_en as score, r.count_europarl7_en as freq 
        ORDER BY score DESC LIMIT $limit
        RETURN distinct(lemposF) AS lemposFs,  freq, score
        '''
    if corpus == "croparl":
        q='''
        MATCH p=(lempos:Lemmas{lempos:$lemma+$pos})-[r:`Lemmas_conj`]-(lemposF:Lemmas) 
        WHERE lemposF.lempos ENDS WITH $pos 
        WITH lemposF.lempos AS lemposF, log(r.freqInCorp) as score, r.freqInCorp as freq 
        ORDER BY score DESC LIMIT $limit
        RETURN distinct(lemposF) AS lemposFs,  freq, score
        '''
    if corpus == "europarl_plenary":
        q='''
        MATCH p=(lempos:Lemma{lempos:$lemma+$pos, language:"en"})-[r:`"%w" and/or ...`]-(lemposF:Lemma) 
        WHERE lemposF.lempos ENDS WITH $pos 
        WITH lemposF.lempos AS lemposF, r.score_europarl_plenary as score, r.count_europarl_plenary as freq 
        ORDER BY score DESC LIMIT $limit
        RETURN distinct(lemposF) AS lemposFs,  freq, score
        '''
         
    df=graph.run(q, lemma=lemma, pos=pos, limit=limit).to_data_frame()
    conumber=len(df)
    print("coocurences for:", lemma, conumber)
    #return (df[['lemposFs', 'score','freq']], conumber)
    
    #Draw the coocurences
    df['Rank'] = ''
    df['Rank'] = np.arange(1, len(df) + 1)
    
    trace1 = go.Scatter(x = df.Rank, y = df.score,
                        mode = "lines+markers", 
                        name = "Score of coocurence",
                        marker = dict(color = 'rgba(16, 112, 2, 0.8)'), 
                        text= df.lemposFs,
                        yaxis='y')
    # Creating trace2
    trace2 = go.Scatter(x = df.Rank, y = df.freq,
                        mode = "lines+markers",
                        name = "Frequency of coocurence",
                        marker = dict(color = 'rgba(80, 26, 80, 0.5)'),
                        text= df.lemposFs,
                        yaxis='y2')
    data = [trace1, trace2]
    layout = go.Layout(title='Score and Frequency of coocurence with: '+lemma+pos+' by Rank', xaxis=dict(title='Rank'), yaxis=dict(title='Score'), yaxis2=dict(title='Frequency', type='log', titlefont=dict(color='rgb(148, 103, 189)'), tickfont=dict(color='rgb(148, 103, 189)'), overlaying='y', side='right'))
    fig = dict(data = data, layout = layout)
    plot(fig)
    pio.write_image(fig, format="svg", width=800, height=400, scale=1.5, file= 'images/score_freq_'+lemma+pos+'_'+corpus+'.svg')


#Usage examples
#and_or_coocurences("dizajn", "-n", "hrwac", 100)
#and_or_coocurences("fear", "-n", "ententen13", 100)
#and_or_coocurences("history", "-n", "europarl7", 100)
#and_or_coocurences("krava", "-n", "croparl", 100)
#and_or_coocurences("fishery", "-n", "europarl_plenary", 100)
