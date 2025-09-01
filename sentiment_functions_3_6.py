#%%
import cgcn_functions_3_6 as cgcn_functions
import numpy as np
from numpy.lib.shape_base import column_stack
import pandas as pd
from senticnet.senticnet import SenticNet
from senticnet.babelsenticnet import BabelSenticNet
import time

#%%############################ SenticNet
############################### Input: single lemma, ex: love
def get_SenticNet_concept(lempos, language):
    # '''for a concept in a language get SenticNet values'''
    if language=='deu':
        language='de'
    if language=='en':
        try:
            sn = SenticNet()
            lemma_info = sn.concept(lempos[0:-2])
        except:
            pass
    else:
        try:
            bsn = BabelSenticNet(language)
            lemma_info = bsn.concept(lempos[0:-2])
        except:
            pass
    return (lempos, lemma_info)#, polarity_label, polarity_value, moodtags, semantics, sentics)
#%%
# ''' output - dictionary:
# {'polarity_label': 'negative',
#  'polarity_value': '-0.88',
#  'sentics': {'introspection': '0', 'temper': '0', 'attitude': '-0.89', 'sensitivity': '-0.86'},
#  'moodtags': ['#disgust', '#fear'],
#  'semantics': ['circuit', 'dismissal', 'needle', 'superfluity', 'redundancy']}
# '''

#%%
def get_SenticNet_concept_df(lempos, language):
    # '''for a concept list in a language get dataframe of SenticNet values'''
    # this is similar as get_SenticNet_concept(concept, language)
    if language=='deu':
        language='de'
    if language=='en':
        try:
            sn = SenticNet()
            lemma_info = sn.concept(lempos.lower()[0:-2])
        except:
            pass
    else:
        try:
            bsn = BabelSenticNet(language)
            lemma_info = bsn.concept(lempos.lower()[0:-2])
        except:
            pass
    df=pd.DataFrame()
    try: 
        df= df.from_dict(lemma_info, orient='index').transpose() #https://stackoverflow.com/questions/40442014/python-pandas-valueerror-arrays-must-be-all-same-length
        #assign sentics dictionary keys to df columns
        df= df.drop(columns='sentics', axis= 1).assign(**pd.DataFrame(df.sentics.values.tolist())) #https://stackoverflow.com/questions/39640936/parsing-a-dictionary-in-a-pandas-dataframe-cell-into-new-row-cells-new-columns   
        #add label
        df['label']= lempos
    except:
        # if no value, just return label
        # if language=='en':
        #     df=pd.DataFrame(columns=['label', 'polarity_label', 'polarity_value', 'introspection', 'temper', 'attitude', 'sensitivity', 'moodtags', 'semantics'])
        # else:
        #     df=pd.DataFrame(columns=['label', 'polarity_label', 'polarity_value', 'pleasantness', 'attention', 'aptitude', 'sensitivity', 'moodtags', 'semantics'])        
        # df= pd.concat([df, pd.Series()], ignore_index=True)
        df= pd.concat([df, pd.Series(dtype='float64')], ignore_index=True) 
        df['label'] = lempos     
    return (df)

# get_SenticNet_concept_df('blood-n', 'en')
# get_SenticNet_concept_df('ženstvenost-n', 'hr')


#%%################################### Input: List of lemmas
def get_SenticNet_c_list(concept_list, language):
    # '''for a concept list in a language get list of SenticNet values'''
    concept_values=[]
    for item in concept_list:
        try:
            concept_values.append(get_SenticNet_concept(item, language))
        except:
            pass
    return concept_values

# get_SenticNet_c_list(['krv-n', 'znoj-n', 'rigatoni-n'], 'hr')

#%%
def get_SenticNet_c_df(concept_list, language):
    # '''for a concept list in a language get dataframe SenticNet values'''
    concept_values=pd.DataFrame()
    for item in concept_list:
        try:
            concept_values= pd.concat([concept_values, get_SenticNet_concept_df(item, language)])
            # concept_values= concept_values.append(get_SenticNet_concept_df(item, language))
        except:
            pass
    return concept_values
# d=get_SenticNet_c_df(['krv-n', 'meso-n', 'urin-n'], 'hr')
# d=get_SenticNet_c_df(['blood-n', 'flesh-n', 'urea-n'], 'en')
# for k in d.keys():
#     if not k in ['label', 'moodtags', 'semantics', 'polarity_label']:
#         print(k)




#%% senticnet srednjica
def get_sentic_srednjica_df(SenticNet_c_df, node_importance_values_list, node_importance_measure):
    # for a dataframe of  sentic values, nodes in a graph, measure of node importance get dataframe of sentic_values 
    # take list of concepts, values, and its node importance and return the middle value
    # SenticNet_c_df je df sa lemama za koju se traži vrijednost, 
    # node_importance_values_list je lista vrijednosti važnosti čvora, 
    # node_importanc_measure je naziv mjere značaja čvora u grafu
    SenticNet_c_df=SenticNet_c_df.dropna()
    
    # Create a dictionary to store the results
    result_dict = {}
    
    for key in SenticNet_c_df.keys(): # for every column create srednjica
        if not key in ['label', 'moodtags', 'semantics']:  # Keep polarity_value and polarity_label
            try:
                # Convert to float and handle any conversion issues
                float_values = []
                for x in SenticNet_c_df[key].tolist():
                    try:
                        float_values.append(float(x))
                    except (ValueError, TypeError) as e:
                        continue
                
                if len(float_values) > 0 and len(node_importance_values_list) > 0:
                    result = cgcn_functions.srednjica(float_values, [float(x) for x in node_importance_values_list])
                    result_dict[key] = result
                    
            except Exception as e:
                continue
    
    # Create DataFrame with centrality measure as a column, not index
    result_dict['centrality_measure'] = node_importance_measure
    return pd.DataFrame([result_dict])

def get_sentic_srednjica_comm(SenticNet_c_df, node_importance_values_list, node_importance_measure):
#     # for a dataframe of  sentic values, nodes in a graph, measure of node importance get dataframe of sentic_values 
#     # take list of concepts, values, and its node importance and return the middle value
#     # SenticNet_c_df je df sa lemama za koju se traži vrijednost, 
#     # node importance_values_list je lista vrijednosti važnosti čvora, 
#     # node_importanc_measure je naziv mjere značaja čvora u grafu
    df=pd.DataFrame()
    for measure in node_importance_measure:
        vdict={}
        for key in SenticNet_c_df.keys(): # for every column create srednjica
            if not key in ['label', 'moodtags', 'semantics']:    # Keep polarity_value and polarity_label
                try:
                    val_list= SenticNet_c_df[key].astype(float).tolist()
                    # Get centrality values for this measure
                    centrality_values = node_importance_values_list[measure].astype(float).tolist()
                    if len(val_list) > 0 and len(centrality_values) > 0:
                        v = cgcn_functions.srednjica(val_list, centrality_values)
                        vdict.update({key:v})
                except Exception as e:
                    continue
        # Add centrality measure as a column
        vdict['centrality_measure'] = measure
        df_measure = pd.DataFrame([vdict])
        df = pd.concat([df, df_measure], ignore_index=True)
    return df



# def get_sentic_srednjica_df(SenticNet_c_df, node_importance_values_list, node_importance_measure):
#     # take list of concepts, values, and its node importance and return the middle value
#     # SenticNet_c_df je df sa lemama za koju se traži vrijednost,
#     # node importance_values_list je lista vrijednosti važnosti čvora,
#     # node_importanc_measure je naziv mjere značaja čvora u grafu
#     SenticNet_c_df=SenticNet_c_df.dropna()
#     sentic_df =pd.DataFrame(index=[node_importance_measure])
#     sentic_df['polarity_value']=cgcn_functions.srednjica([float(x) for x in SenticNet_c_df['polarity_value']], node_importance_values_list )
#     sentic_df['introspection']=cgcn_functions.srednjica([float(x) for x in SenticNet_c_df['introspection']], node_importance_values_list )
#     sentic_df['temper']=cgcn_functions.srednjica([float(x) for x in SenticNet_c_df['temper']], node_importance_values_list )
#     sentic_df['attitude']=cgcn_functions.srednjica([float(x) for x in SenticNet_c_df['attitude']], node_importance_values_list )
#     sentic_df['sensitivity']=cgcn_functions.srednjica([float(x) for x in SenticNet_c_df['sensitivity']], node_importance_values_list )
#     return sentic_df



def get_graph_sentic_values(SenticNet_c_df, graph_df):
    # for a dataframe of  sentic values, nodes in a graph get dataframe of sentic_values according to a listOf node_importance_measures
    node_importance_measures=['sli', 'pagerank', 'degree', 'weighted_degree', 'betweenness']
    graph_sentic_values = pd.DataFrame()
    
    try:
        for node_importance_measure in node_importance_measures: 
            
            # get node importance from graph_DataFrame for all labels containing some value in sentic_concepts_df
            labels_with_polarity = SenticNet_c_df[SenticNet_c_df['polarity_value'].notna()]['label']
            
            # Convert to string for comparison
            labels_with_polarity_str = [str(x) for x in labels_with_polarity]
            
            # Check which labels exist in graph_df
            matching_labels = graph_df[graph_df["label"].isin(labels_with_polarity_str)]
            
            if len(matching_labels) == 0:
                print(f"WARNING: No matching labels found for measure {node_importance_measure}")
                continue
                
            # Get centrality values for matching labels
            node_importance_values_list = matching_labels[node_importance_measure].astype('float').tolist()
            
            if len(node_importance_values_list) == 0:
                print(f"WARNING: No centrality values found for measure {node_importance_measure}")
                continue
                
            # Calculate GSV
            gsv_result = get_sentic_srednjica_df(SenticNet_c_df, node_importance_values_list, node_importance_measure)
            
            # The centrality_measure is already added in get_sentic_srednjica_df
            graph_sentic_values = pd.concat([graph_sentic_values, gsv_result], ignore_index=True)
            
    except Exception as e:
        print(f"ERROR in get_graph_sentic_values: {str(e)}")
        import traceback
        traceback.print_exc()
        pass
    
    return (graph_sentic_values)



#%%
def make_sentic_df(sentic_list, node_importance_values_list, node_importance_measure, language):
    try:
        # take list of concepts, values, and its node importance and return the middle value
        # sentic_list je lista lema za koju se traži vrijednost, 
        # node importance_values_list je lista vrijednosti važnosti čvora, 
        # node_importanc_measure je naziv mjere značaja čvora u grafu
        sentic_df =pd.DataFrame(index=[node_importance_measure])
        if language=='en':
            sentic_df['polarity_value']=cgcn_functions.srednjica([float(x[1]['polarity_value']) for x in sentic_list], node_importance_values_list )
            sentic_df['introspection']=cgcn_functions.srednjica([float(x[1]['sentics']['introspection']) for x in sentic_list], node_importance_values_list )
            sentic_df['temper']=cgcn_functions.srednjica([float(x[1]['sentics']['temper']) for x in sentic_list], node_importance_values_list )
            sentic_df['attitude']=cgcn_functions.srednjica([float(x[1]['sentics']['attitude']) for x in sentic_list], node_importance_values_list )
            sentic_df['sensitivity']=cgcn_functions.srednjica([float(x[1]['sentics']['sensitivity']) for x in sentic_list], node_importance_values_list )
        else:            
            sentic_df['polarity_value']=cgcn_functions.srednjica([float(x[1]['polarity_value']) for x in sentic_list], node_importance_values_list )
            sentic_df['pleasantness']=cgcn_functions.srednjica([float(x[1]['sentics']['introspection']) for x in sentic_list], node_importance_values_list )
            sentic_df['attention']=cgcn_functions.srednjica([float(x[1]['sentics']['temper']) for x in sentic_list], node_importance_values_list )
            sentic_df['aptitude']=cgcn_functions.srednjica([float(x[1]['sentics']['aptitude']) for x in sentic_list], node_importance_values_list )
            sentic_df['sensitivity']=cgcn_functions.srednjica([float(x[1]['sentics']['sensitivity']) for x in sentic_list], node_importance_values_list )
    except:
        pass
    finally:
        pass
    return sentic_df


#%%
def calculate_sentic_values(listOfLemmas, pos, language, measure, corpus, corpusID, gramRel, limit_snp_f, limit_snp_fof, harvestSelect):
    # for a lemma in df_senticnet get sourceValuePropagation 2- napraviti izračun po metodi - sourceValuePropagation: val(sVP)
    # pos = -n, -v, -j
    # start=time.time() # measure time
    sentic_calculated_values=pd.DataFrame() # create a DataFrame
    for lemma in listOfLemmas: 
        lemma= lemma+pos # assign a pos to search in corpus lempos
        if not '_' in lemma: #exclude based on _
            try:
                # f_snp_friends
                f_snp_df = cgcn_functions.friends_df(language, lemma[0:-2], lemma[-2:], corpus, corpusID, gramRel, lemma[-2:], limit_snp_f, measure, harvestSelect, 'undirected')
                # fof_snp_friends
                fof_snp_df= cgcn_functions.FoFData(language, f_snp_df, lemma[0:-2], lemma[-2:], corpus, corpusID, gramRel, lemma[-2:], limit_snp_fof, measure, harvestSelect, 'undirected')
                fof_snp_graph = cgcn_functions.FoF_graph(fof_snp_df, measure)
                fof_snp_graph_df = cgcn_functions.df_from_graph(fof_snp_graph)
                # get sentic values for lexical nodes in FoF
                sentic_snp = get_SenticNet_c_df(fof_snp_graph.vs["label"], language)
                # get sentic value for a lemma in dictionary based on the  'sentic_snp '+lemma, sentic_snp
                sentic_snp_value = get_graph_sentic_values(sentic_snp, fof_snp_graph_df)
                sentic_snp_value['label'] = lemma
                # sentic_calculated_values = pd.concat([sentic_calculated_values, sentic_snp_value])
                sentic_calculated_values = sentic_calculated_values.append(sentic_snp_value)
            except:
                pass
        else:
            pass
    return sentic_calculated_values


# calculate_sentic_values(['corona'], pos, language, measure, corpus, corpusID, gramRel, limit_snp_f, limit_snp_fof, harvestSelect)
########################################## Vader
def get_vader_srednjica_df(vader_values_df, node_importance_values_list, node_importance_measure):
    # take list of concepts, values, and its node importance and return the middle value
    # sentiWords_values_df je df sa lemama za koju se traži vrijednost, 
    # node importance_values_list je lista vrijednosti važnosti čvora, 
    # node_importanc_measure je naziv mjere značaja čvora u grafu
    vader_values_df=vader_values_df.dropna()
    vader =pd.DataFrame(index=[node_importance_measure])
    vader['compound']=cgcn_functions.srednjica([float(x) for x in vader_values_df['compound']], node_importance_values_list)
    return vader



#%%#%%##################################### SentiWords 1.1
# import sentiWords as dataframe
sentiWords = pd.read_csv('SentiWords_1.1.txt', header=25, sep='\t')
sentiWords['lempos']= sentiWords['lemma']+'-'+sentiWords['pos'] 
# keys: 'lemma', 'pos', 'polarity_score'

#%%
# sentiWords[sentiWords['lempos']=='karlovac-n']
#%%
def get_sentiWords_values_df(lempos_list, sentiWordsDict):
    #for a list of lempos get values from sentiWords df
    df= sentiWordsDict[sentiWordsDict['lempos'].isin(lempos_list)][['lempos', 'polarity_score']]
    return df
# get_sentiWords_values_df(['fear-n', 'love-n', 'blood-n'], sentiWords)

def get_sentiWords_srednjica_df(sentiWords_values_df, node_importance_values_list, node_importance_measure):
    # take list of concepts, values, and its node importance and return the middle value
    # sentiWords_values_df je df sa lemama za koju se traži vrijednost, 
    # node importance_values_list je lista vrijednosti važnosti čvora, 
    # node_importanc_measure je naziv mjere značaja čvora u grafu
    sentiWords_values_df=sentiWords_values_df.dropna()
    sentiWords =pd.DataFrame(index=[node_importance_measure])
    sentiWords['polarity_score']=cgcn_functions.srednjica([float(x) for x in sentiWords_values_df['polarity_score']], node_importance_values_list)
    return sentiWords

#%%
def get_sentiWords_values_graph_df(sentiWords_values_df, graph_df ):
    #get sentiWords values for a nodes in a graph
    node_importance_measures=['pagerank', 'degree', 'weighted_degree', 'betweenness', 'sli']
    graph_sentiWords_values = pd.DataFrame()
    for node_importance_measure in node_importance_measures: 
        # get node importance from graph_DataFrame for all labels containing some value in sentiWords_concepts_df
        node_importance_values_list = graph_df[graph_df["label"].isin([str(x) for x in sentiWords_values_df[sentiWords_values_df['polarity_score'].notna()]['lempos']])][node_importance_measure].astype('float').tolist()
        graph_sentiWords_values= pd.concat([graph_sentiWords_values, get_sentiWords_srednjica_df(sentiWords_values_df, node_importance_values_list, node_importance_measure)])
    return (graph_sentiWords_values)
      
def calculate_sentiWords_values(sentiWordsDict, listOfLemmas, pos, language, measure, corpus, corpusID, gramRel, limit_snp_f, limit_snp_fof, harvestSelect):
    # for a lemma in df_senticnet get sourceValuePropagation 2- napraviti izračun po metodi - sourceValuePropagation: val(sVP)
    # pos = -n, -v, -j
    # start=time.time() # measure time
    sentiWords_calculated_values=pd.DataFrame() # create a DataFrame
    for lemma in listOfLemmas: 
        # lemma= lemma+pos # assign a pos to search in corpus lempos
        try:
            # f_snp_friends
            f_snp_df = cgcn_functions.friends_df(language, lemma, pos, corpus, corpusID, gramRel, pos, limit_snp_f, measure, harvestSelect, 'undirected')
            # fof_snp_friends
            fof_snp_df= cgcn_functions.FoFData(language, f_snp_df, lemma, pos, corpus, corpusID, gramRel, pos, limit_snp_fof, measure, harvestSelect, 'undirected')
            fof_snp_graph = cgcn_functions.FoF_graph(fof_snp_df, measure)
            fof_snp_graph_df = cgcn_functions.df_from_graph(fof_snp_graph)
            # get sentiWords values for lexical nodes in FoF
            sentiWords_snp = get_sentiWords_values_df(fof_snp_graph.vs["label"], sentiWordsDict)
            # get sentic value for a lemma in dictionary based on the  'sentiWords_snp '+lemma, sentiWords_snp
            sentiWords_snp_value = get_sentiWords_values_graph_df(sentiWords_snp, fof_snp_graph_df)
            sentiWords_snp_value['label'] = lemma
            sentiWords_calculated_values = pd.concat([sentiWords_calculated_values, sentiWords_snp_value])
        except:
            pass
        
    return sentiWords_calculated_values




#%%
sentiWordsNet = pd.read_csv('SentiWordNet.txt', sep='\t')
# sentiWordsNet['ID']=sentiWordsNet['ID'].astype(int)
# sentiWordsNet.head()


def get_word_sentiWordsNet_values_df(sentiWordsNet, lemposlist):
    df= pd.DataFrame()
    for lempos in lemposlist:
        try:
            lemma= lempos[0:-2]
            pos= lempos[-1:]
            lemmaSynTerm=lemma+"#"
            query= sentiWordsNet[(sentiWordsNet['SynsetTerms'].str.match(lemmaSynTerm)) & (sentiWordsNet['POS']==pos)]
            df= pd.concat([df,query])
        except:
            pass
    return df
#%%
# def get_word_sentiWordsNet_values_df(sentiWordsNet, lemposlist):

# get_word_sentiWordsNet(sentiWordsNet, 'fear', 'n')




