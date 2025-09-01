#%% 
import numpy as np
import pandas as pd
import nltk
# nltk.download('omw')
from nltk.corpus import wordnet as wn
from itertools import islice
#%%
def lang_modification(lang):
    '''correct lang value due to the difference in sketch2Neo vs wordnet requeirements in language metadata'''
    if lang=='hr':
        lang='hrv'
    if lang=='it':
        lang='ita'
    if lang=='en':
        lang= 'eng'
    if lang== 'deu':
        lang='de'
    return (lang)
# %% get_synset_from_lempos and language
def get_synset_from_lempos(lemposList, lang):
    lang=lang_modification(lang)
    synsetList=[]
    for lempos in lemposList:
        lemma=lempos.split('-')[0]
        pos=lempos.split('-')[1]
        if pos=='j':
            pos='a'    
        for lemma in wn.lemmas(lemma, lang=lang):
            try:
                synsetList.append(lemma.synset())
            except:
                pass
    ds=pd.Series(synsetList)
    return (ds)

# get_synset_from_lempos(['more-n'], 'hrv')
#%% get synset and hypernym from a lempos list
def get_hypernyms(lemposList, lang):
    lang=lang_modification(lang)
    commonList=[]
    itemslist=[]
    for lempos in lemposList:
        lemma=lempos.split('-')[0]
        pos=lempos.split('-')[1]
        if pos=='j':
            pos='a'
        # get all synsets for every lemma according to language
        synsetList=[]
        try:
            for lemma in wn.lemmas(lemma, pos=pos, lang=lang):
                synsetList.append(lemma.synset()) # appends a synset to a list
            
            # get hypernyms for item from a synsetList
            for item in synsetList:
                try:
                    hyp= item.hypernyms()
                    for h in hyp:
                        tup = (item, h, lempos)
                        commonList.append(tup)
                        
                except:
                    pass
        except:
            pass
    df=pd.DataFrame(commonList, columns=[ 'source', 'target', 'lempos'])
    return df
#%%
# get_hypernyms(['emozione-n'], 'ita')
#%%
def get_lowest_common_hypernyms(labelList):
    commonList=[]
    for item in labelList:
        lemma=item.split('-')[0]
        pos=item.split('-')[1]

        # get all synsets for every lemma
        synsetList=[]
        for synset in wn.synsets(lemma, pos=pos):
            synsetList.append(synset) # appends a synset to a list
        
        # get lowest common hypernyms for item1 and item2 from a synsetList
        for item in synsetList:
            for item2 in synsetList:
                if not str(item).split('.')[0] == str(item2).split('.')[0]: #do not use same lemma synsets
                    c_h= item.lowest_common_hypernyms(item2)[0]
                    triangle= (item,item2, c_h)
                    commonList.append(triangle)
    df=pd.DataFrame(commonList, columns=['source1', 'source2', 'target'])
    return df

# %%
def get_definition(syn):
    definition = wn.synset(syn).definition()
    return definition
#%% sentiment analysis using NLTK Vader
# https://towardsdatascience.com/sentimental-analysis-using-vader-a3415fef7664
# nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
#%%
def get_valency_for_lempos(lemposList):
    words = ', '.join(', '.join([x.split('.')[0] for x in lemposList]).split('_'))
    #print(words)
    return (sid.polarity_scores(words))

# get_valency_for_lempos(['love.n'])

#%%
def get_wnValency_for_lempos_df(lemposList, language):
    dct= []
    for word in lemposList:
        dct.append(sid.polarity_scores(word[0:-2]))
    df=pd.DataFrame.from_dict(dct)
    df['label']=lemposList
    return df

# def get_wnValency_srednjica_df(SenticNet_c_df, node_importance_values_list, node_importance_measure):
    # take list of word, values, and its node importance and return the middle value
    # SenticNet_c_df je df sa lemama za koju se traži vrijednost, 
    # node importance_values_list je lista vrijednosti važnosti čvora, 
    # node_importanc_measure je naziv mjere značaja čvora u grafu
    # SenticNet_c_df=SenticNet_c_df.dropna()
    # sentic_df =pd.DataFrame(index=[node_importance_measure])
    # sentic_df['polarity_value']=cgcn_functions.srednjica([float(x) for x in SenticNet_c_df['polarity_value']], node_importance_values_list )
    # sentic_df['introspection']=cgcn_functions.srednjica([float(x) for x in SenticNet_c_df['introspection']], node_importance_values_list )
    # sentic_df['temper']=cgcn_functions.srednjica([float(x) for x in SenticNet_c_df['temper']], node_importance_values_list )
    # sentic_df['attitude']=cgcn_functions.srednjica([float(x) for x in SenticNet_c_df['attitude']], node_importance_values_list )
    # sentic_df['sensitivity']=cgcn_functions.srednjica([float(x) for x in SenticNet_c_df['sensitivity']], node_importance_values_list )
    # return sentic_df


#     words = ', '.join(', '.join([x.split('.')[0] for x in lemposList]).split('_'))
#     #print(words)
#     return (sid.polarity_scores(words))

#%%
#%%
def get_lemma_from_synset(synset):
    # for lemma in wn.synset('sea.n').lemmas():
    #     print(lemma, lemma.count())
    return wn.synset(synset).lemmas().lemma
# get_lemma_from_synset('familiarity.n.01')

#%% antonyms
def get_antonyms_from_lempos(lemposList, lang):
    lang=lang_modification(lang)
    synsetList=[]
    antonymsList=[]
    antonyms_all= [] 
    
    #get synsetlist from lempos
    for lempos in lemposList:
        lemma=lempos[:-2]
        pos=lempos[-2:]
        if pos=='j':
            pos='a'    
        for lemma in wn.lemmas(lemma, lang=lang):
            try:
                synsetList.append(lemma.synset())
            except:
                pass
        # get antonyms_sinset list from synsets
        for ss in synsetList:
            for lemma in ss.lemmas():
                any_pos_antonyms = [ antonym.name() for antonym in lemma.antonyms() ]
                antonymsList.append(any_pos_antonyms)
        # get antonym lemma from antoyms
        for lista in antonymsList:
            for ant in lista:
                for ant_syn in wn.synsets(ant): 
                    for ant_l in ant_syn.lemmas(lang=lang): 
                        antonyms_all.append({'lempos':lempos, 'antonym': ant_l})
        
  

    if antonyms_all:
        ds_ant=pd.DataFrame.from_dict(antonyms_all)
        ds_ant['ant_lempos']= [x[-1][:-2]+'-'+x[-3] for x in ds_ant['antonym'].astype(str).str.split('.')]
        ds_ant['ant_wordnet_lemma'] = [x[0][7:]+'.'+x[1]+'.'+x[2] for x in ds_ant['antonym'].astype(str).str.split('.')]
        ds_ant=ds_ant[['lempos', 'ant_wordnet_lemma', 'ant_lempos']].drop_duplicates()
    else:
        ds_ant= None
    return (ds_ant)


# %%
