# -*- coding: utf-8 -*-

# =============================================================================
# Script for storing WordSketches obtained from SketchEngine API.
# Create indexes.
# Set the time between requests (2-4 seconds)
# Functions for each corpus mapped on to the structure of the graph database.
# =============================================================================

from py2neo import Graph, DatabaseError
import requests
import json
import pandas as pd
from time import sleep
import authSettings as aS # authentication settings are stored here
userName =aS.userName
apiKey = aS.apiKey



#Set time between queries
sleepTime=2

#MaxItems - set the amount of initial harvested collocations
maxitems=300

metoda='wsketch?'

# =============================================================================
# Connecting to the Neo4j Database
# =============================================================================
# Define a function to connect to the database
def connect_to_database():
    try:
        #return Graph("bolt://193.198.209.95:7687", auth=(aS.graphUser, aS.graphPass))
        return Graph("bolt://localhost:7687", auth=(aS.graphUser, aS.graphPass))
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

graph = connect_to_database()


# try:
#     #local
#     # graph = Graph("bolt://localhost:7687", auth=(aS.graphUser, aS.graphPass))   
#     # vps
#     # graph = Graph("bolt://31.147.204.249:7687", auth=(aS.graphUser, aS.graphPass))
#     graph = Graph("bolt://congracnet.uniri.hr:7687", auth=(aS.graphUser, aS.graphPass))
# except:
#     pass    

# =============================================================================
# Setting up indexes and constraints the Neo4j database
# =============================================================================
def createIndexes():
    # Better checks for each specific index
    index_queries = [
        #"CREATE INDEX ON :Lemma(name)",
        #"CREATE INDEX ON :Lemma(lempos)",
        #"CREATE INDEX ON :Lemma(language)"
        "CREATE INDEX IF NOT EXISTS FOR (n:Lemma) ON (n.name);",
        "CREATE INDEX IF NOT EXISTS FOR (n:Lemma) ON (n.lempos);",
        "CREATE INDEX IF NOT EXISTS FOR (n:Lemma) ON (n.language);"
    ]
    
    for query in index_queries:
        try:
            graph.run(query)
        except DatabaseError:
            # This means index probably exists, so we can pass
            pass
        except Exception as e:
            print(f"Error creating index with query {query}: {e}")

createIndexes() 

# def createIndexes():
#     #check if the Indexes already exist
#     queryCheckIndex="""
#     CALL db.indexes() 
#     """
#     n= graph.run(queryCheckIndex).to_data_frame()
#     print(n['description'])
#     if n['description'].empty:
#         print('Creating Indexes:')    
#         try:
#             query1="""
#             CREATE INDEX ON :Lemma(name)
#             """
#             graph.run(query1)
#             query2="""
#             CREATE INDEX ON :Lemma(lempos)
#             """
#             graph.run(query2)
#             query3="""
#             CREATE INDEX ON :Lemma(language)
#             """
#             graph.run(query3) 

#             print("Indexes created")
#         except:
#             pass
        
# =============================================================================
# Neo4j summary
# =============================================================================
def neo4jSummaryLemma():
    '''count all lemmas and all relationships'''
    try:
        query = """
        MATCH (n:Lemma)
        WITH count(n) as lemmaCount
        MATCH ()-[r]->() 
        WITH count(r) as relsCount, lemmaCount
        RETURN lemmaCount, relsCount
        """
        d = graph.run(query).to_data_frame()
        lemmaCount = d.lemmaCount[0]
        relsCount = d.relsCount[0]
        return (int(lemmaCount), int(relsCount))
    except Exception as e:
        print(f"Error executing neo4jSummaryLemma: {e}")
        return (None, None)  # or consider returning (0, 0) if that makes more sense in your application


# def neo4jSummaryLemma():
#     '''count all lemmas and all relationships'''
#     query="""
#     MATCH (n:Lemma)
#     WITH count(n) as lemmaCount
#     MATCH ()-->() 
#     WITH count(*) as relsCount, lemmaCount
#     RETURN lemmaCount, relsCount
#     """
#     d=graph.run(query).to_data_frame()
#     lemmaCount= d.lemmaCount[0]
#     relsCount = d.relsCount[0]
#     return(int(lemmaCount), int(relsCount))


# =============================================================================
# Corpus selection
# To add another corpus create new if statement in:
#  1-corpusSelect()
#  2-corpusSketchHarvest()
#  3- cS list in cgcnStream.py 
#  4- create new df wordSketch_x()
# =============================================================================

# define: cS, corpus, language, corpusID, gramRel (coordination)
corpusList = (
    {'cS': "ententen13_tt2_1 (English Web)", 'corpus': "preloaded/ententen13_tt2_1", 'corpusID': "ententen13_tt2_1", 'language':'en', 'gramRel':'"%w" and/or ...', 'initLemma':'emotion','initPos':'n'}, 
    {'cS': "europarl7_en (English EuroParlamentary)", 'corpus': "preloaded/europarl7_en", 'corpusID': "europarl7_en", 'language':'en', 'gramRel':'and/or', 'initLemma':'emotion','initPos':'n'}, 
    {'cS': 'europarl_plenary (English Plenary sessions of EuroParliament)', 'corpus': "user/benediktperak/europarl_plenary", 'corpusID':"europarl_plenary", 'language':'en', 'gramRel':'"%w" and/or ...', 'initLemma':'emotion','initPos':'n'}, 
    {'cS':'hrWac22 (hrWac22_ws Croatian Web)', 'corpus': 'preloaded/hrwac22_ws', 'corpusID':"hrwac22", 'language': 'hr', 'gramRel':'koordinacija', 'initLemma':'emocija','initPos':'n'}, 
    {'cS':'ittenten16_2 (Italian Web Corpus 5,8 GW)', 'corpus': 'preloaded/ittenten16_2', 'corpusID':"ittenten16_2", 'language': 'it', 'gramRel':'"%w" and/or ...', 'initLemma':'emozione','initPos':'n'}, 
    {'cS':'Timestamped JSI web corpus 2014-2019 English', 'corpus': 'preloaded/eng_jsi_newsfeed_virt', 'corpusID':"eng_jsi_newsfeed_virt", 'language': 'en', 'gramRel': '"%w" and/or ...', 'initLemma':'emotion','initPos':'n'},
    {'cS':'Timestamped JSI web corpus 2020-06 English', 'corpus': 'preloaded/eng_jsi_newsfeed_lastmonth', 'corpusID':"eng_jsi_newsfeed_lastmonth", 'language': 'en', 'gramRel': '"%w" and/or ...', 'initLemma':'emotion','initPos':'n', 'info':'English corpus of news articles obtained from crawled a list of RSS feeds. Corpus tagged by TreeTagger v2.'},
    {'cS': 'Timestamped JSI web corpus 2014-2020 German', 'corpus': 'preloaded/deu_jsi_newsfeed_virt', 'corpusID':'deu_jsi_newsfeed_virt', 'language':'deu', 'gramRel': '"%w" and/or ...', 'initLemma':'emotion','initPos':'n',},  
)       
corpusListDf= pd.DataFrame(corpusList)



# def corpusSelect(cS): 
#     corpusSelected = corpusListDf[corpusListDf['cS'] == cS]
    
#     if corpusSelected.empty:
#         print(f"No corpus found for the given cS: {cS}")
#         return None
    
#     # Fetch the first row of the filtered dataframe as a dictionary.
#     # corpus_dict = corpusSelected['cS', 'corpus', 'language', 'corpusID', 'gramRel', 'initLemma', 'initPos', 'info'].
    
#     # Return the values of the dictionary as a tuple.
#     return corpus_dict


def corpusSelect(cS): 
    corpusSelected = corpusListDf[corpusListDf['cS']== cS]
    if corpusSelected.empty:
        print(f"No corpus found for the given cS: {cS}")
        return None
    cS = corpusSelected["cS"].values[0]
    corpus = corpusSelected["corpus"].values[0]
    corpusID= corpusSelected['corpusID'].values[0]
    language= corpusSelected['language'].values[0]
    gramRel= corpusSelected['gramRel'].values[0]
    initLemma= corpusSelected['initLemma'].values[0] 
    initPos= corpusSelected['initPos'].values[0] 
    info = corpusSelected['info'].values[0]  
    return(cS, corpus, language, corpusID, gramRel, initLemma, initPos, info)

# #%% corp info try
# import requests
# url= "https://api.sketchengine.eu/bonito/run.cgi/corp_info?corpname=bnc2;gramrels=1;subcorpora=1&username="+userName+"&api_key="+apiKey
# requests.get(url).text


def corpus_info(corpusID, userName, apiKey):
    url = f"https://api.sketchengine.eu/bonito/run.cgi/corp_info?&username={userName}&api_key={apiKey}&corpname={corpusID};gramrels=1;subcorpora=1"
    
    try:
        response = requests.get(url)
        response.raise_for_status() # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching corpus info: {e}")
        return None


# def corpus_info(corpusID, userName, apiKey):
#     q='''
#     WITH ("https://api.sketchengine.eu/bonito/run.cgi/corp_info?&username="+$userName+"&api_key="+$apiKey+"&corpname="+$corpusID+";gramrels=1;subcorpora=1") AS url
#         CALL apoc.load.json(url) YIELD value
#     return value
#     '''
#     result=graph.run(q,corpusID=corpusID, userName=userName, apiKey=apiKey).to_data_frame()
#     return (result)


# =============================================================================
# List all grammar relations from a corpus
# =============================================================================
# def listGramRel(corpusID):
#     q=""" MATCH (n:Lemma)-[r]->(m:Lemma) WHERE r[$count]<>"" 
#     RETURN distinct(type(r)) as gramRels
#     """
#     df= graph.run(q, count='count_'+corpusID).to_data_frame()    
#     return (df)

def listGramRel(corpusID):
    q = """
    MATCH (n:Lemma)-[r]->(m:Lemma)
    WHERE r[count_property] <> ""
    RETURN distinct type(r) as gramRels
    """
    
    # Pass the constructed property name to the query
    df = graph.run(q, count_property='count_' + corpusID).to_data_frame()
    
    return df


# =============================================================================
# Corpus selection dependent activation of WordSketch harvesting
# =============================================================================
# def corpusSketchHarvest(cS, lemmeOdabrane, userName, apiKey, metoda, pos, pos2, corpus, maxitems, language, gramRel):
#     try:
#         print("WordSketches for lempos: ", str(list(lemmeOdabrane)).replace("[","").replace("]","").replace("'","")+str(pos), "maxItems=",maxitems, "in", "https://app.sketchengine.eu/#dashboard?corpname="+str(corpus))
#         if cS== "ententen13_tt2_1 (English Web)":         
#             wordSketchEnTenTen13(lemmeOdabrane, userName, apiKey, metoda, pos, pos2, corpus, maxitems, language, gramRel)    
#         if cS == "europarl7_en (English EuroParlamentary)":
#             wordSketchEuroParl7(lemmeOdabrane, userName, apiKey, metoda, pos, pos2, corpus, maxitems, language, gramRel)
#         if cS == 'europarl_plenary (English Plenary sessions of EuroParliament)':
#             wordSketchEuroParlPlenary(lemmeOdabrane, userName, apiKey, metoda, pos, pos2, corpus, maxitems, language, gramRel)
#         if cS == 'hrWac22 (hrWac22_ws Croatian Web)':
#             wordSketchHrWac22(lemmeOdabrane, userName, apiKey, metoda, pos, pos2, corpus, maxitems, language, gramRel)
#         if cS == 'ittenten16_2 (Italian Web Corpus 5,8 GW)':
#             wordSketchIttenten16(lemmeOdabrane, userName, apiKey, metoda, pos, pos2, corpus, maxitems, language, gramRel)
#         if cS == 'Timestamped JSI web corpus 2014-2019 English':
#             wordSketchEng_jsi_newsfeed_virt(lemmeOdabrane, userName, apiKey, metoda, pos, pos2, corpus, maxitems, language, gramRel)
#         if cS == 'Timestamped JSI web corpus 2020-06 English':
#             wordSketchEng_jsi_newsfeed_lastmonth(lemmeOdabrane, userName, apiKey, metoda, pos, pos2, corpus, maxitems, language, gramRel)
#         if cS == 'Timestamped JSI web corpus 2014-2020 German':
#             wordSketchDeu_jsi_newsfeed_virt(lemmeOdabrane, userName, apiKey, metoda, pos, pos2, corpus, maxitems, language, gramRel)
#     except:
#         pass
#     return(corpus)


def corpusSketchHarvest(cS, lemmeOdabrane, userName, apiKey, metoda, pos, pos2, corpus, maxitems, language, gramRel):
    # Dictionary for corpus function dispatch
    corpus_dispatch = {
        "ententen13_tt2_1 (English Web)": wordSketchEnTenTen13,
        "europarl7_en (English EuroParlamentary)": wordSketchEuroParl7,
        "europarl_plenary (English Plenary sessions of EuroParliament)": wordSketchEuroParlPlenary,
        "hrWac22 (hrWac22_ws Croatian Web)": wordSketchHrWac22,
        "ittenten16_2 (Italian Web Corpus 5,8 GW)": wordSketchIttenten16,
        "Timestamped JSI web corpus 2014-2019 English": wordSketchEng_jsi_newsfeed_virt,
        "Timestamped JSI web corpus 2020-06 English": wordSketchEng_jsi_newsfeed_lastmonth,
        "Timestamped JSI web corpus 2014-2020 German": wordSketchDeu_jsi_newsfeed_virt
    }
    
    # Print and call the appropriate function
    try:
        print("WordSketches for lempos: ", str(list(lemmeOdabrane)).replace("[","").replace("]","").replace("'","") + str(pos),
              "maxItems=", maxitems, "in", "https://app.sketchengine.eu/#dashboard?corpname=" + str(corpus))
        
        # Call the appropriate function using the dictionary
        if cS in corpus_dispatch:
            corpus_dispatch[cS](lemmeOdabrane, userName, apiKey, metoda, pos, pos2, corpus, maxitems, language, gramRel)
        else:
            print(f"Unknown corpus: {cS}")
            return None
    except Exception as e:
        print(f"Error processing corpus {cS}: {e}")
    
    return corpus

# =============================================================================
# sketch2neo EnTenTen13 wordSketch harvest
# =============================================================================
def wordSketchEnTenTen13(lemmeOdabrane, userName, apiKey, metoda, pos, pos2, corpus, maxitems, language, gramRel):
    for lemmaOdabrana in lemmeOdabrane:
        query="""       
        //trazenje rijeci u gramrel: wsketch? metoda
        WITH ("https://api.sketchengine.eu/bonito/run.cgi/"+
        $metoda+"username="+$userName+"&api_key="+$apiKey+
        "&corpname="+$corpus+"&lemma="+$lemmaOdabrana+"&lpos="+$pos+"&maxitems="+$maxitems+"&format=json") AS url
        CALL apoc.load.json(url) YIELD value
        
        // send request YIELD value.lemma > ako ne dobije podatke sa SketchEngin Corpusa završava upit ELSE ide dalje s procedurom MERGE
        FOREACH (x IN CASE WHEN value.lemma IS NULL THEN [] ELSE [1] END | 
            MERGE (from:Lemma{lempos:value.lemma+value.lpos, language:$language})
            ON CREATE SET from.name=value.lemma
            ON MATCH SET from.lpos=value.lpos
            ON MATCH SET from.corpus=value.corp_full_name
            ON MATCH SET from.freq_ententen13_tt2_1=toInteger(value.freq)
            ON MATCH SET from.relFreq_ententen13_tt2_1=toFloat(value.relfreq)
            ON MATCH SET from.primjerLink_ententen13_tt2_1='https://api.sketchengine.eu/bonito/run.cgi/view?'+value.lemma_conc_link+';username='+$userName+'&api_key='+$apiKey+'&corpname='+$corpus //đđ ovo je dobar link sa parametrima
            ON CREATE SET from.language=$language)
        
            //gramatičke relacije pretvorene u nodes
        WITH value as Value
            //nađi u bazi lemu s kojim će se povezivati preko lemposa koji su unique + jezik
        MATCH (lemma:Lemma{lempos:$lemmaOdabrana+$pos, language:$language})
        UNWIND Value.Gramrels as Gramrels
        
            //napravi vezu: lemma ima GrammaticRelations
        FOREACH (
        grels in Gramrels | MERGE (grel:GrammaticRelations{name:Gramrels.name}) ON CREATE SET grel.language=$language
            MERGE (lemma)-[r:HAS_GRAMREL]->(grel)
                ON CREATE SET r.seek_ententen13_tt2_1=Gramrels.seek, r.count=Gramrels.count, r.score=Gramrels.score
                ON CREATE SET r.corpus_ententen13_tt2_1=r.corpus+$corpus
                ON CREATE SET r.GramRelPrimjerLink_ententen13_tt2_1='https://api.sketchengine.eu/bonito/run.cgi/view?q=w'+Gramrels.seek+';username=bperak&api_key=VZSUO4L8NV5Y9WVNWNRFU2JRRXKGTK3E&corpname='+{corpus}
        )
        
            //to lemma
        WITH Gramrels AS gramrelSelected
        UNWIND gramrelSelected.Words AS Words
        
            //stvori to lemme iz WSketcha i poveži s lemmafrom
        FOREACH (
        word IN CASE WHEN Words.lempos <> "" THEN [1] ELSE [] END  | 
            MERGE (to:Lemma{lempos:Words.lempos, language:$language}) ON CREATE SET to.name=Words.word
        )
        WITH gramrelSelected, Words
        MATCH (to:Lemma{lempos:Words.lempos, language:$language})
        MATCH(lemmafrom:Lemma{lempos:$lemmaOdabrana+$pos, language:$language})
        WITH lemmafrom, to, gramrelSelected, Words
        
        CALL apoc.merge.relationship(lemmafrom, gramrelSelected.name,{relID:Words.id},{type:gramrelSelected.name, count_ententen13_tt2_1:Words.count,score_ententen13_tt2_1:Words.score, cm_ententen13_tt2_1:Words.cm, seek_ententen13_tt2_1:Words.seek, relID:Words.id, exampleLink_ententen13_tt2_1:'https://api.sketchengine.eu/bonito/run.cgi/view?q=w'+Words.seek+';username=bperak&api_key=VZSUO4L8NV5Y9WVNWNRFU2JRRXKGTK3E&corpname='+{corpus},language:$language}, to) YIELD rel
        RETURN lemmafrom.name as From, type(rel), to.name as To, rel.count_ententen13_tt2_1 as Freq, rel.score_ententen13_tt2_1 as Sq 
        ORDER BY type(rel), Freq, Sq DESC
        """
        graph.run(query, userName=userName, apiKey=apiKey, metoda=metoda, lemmaOdabrana=lemmaOdabrana, pos=pos, corpus=corpus, maxitems=maxitems, language=language) #parametri
    
        for lemmaOdabrana in lemmeOdabrane:
            query2="""
            MATCH (n:Lemma{lempos:$lemmaOdabrana+$pos,language:$language})-[r]-(m:Lemma) 
            WHERE m.lempos ENDS WITH $pos2 AND type(r)=$gramRel AND NOT EXISTS (m.freq_ententen13_tt2_1)
            RETURN m.name AS lemmaNova
            """
            data2 = graph.run(query2, userName=userName, apiKey=apiKey, lemmaOdabrana=lemmaOdabrana, pos=pos, pos2=pos2, gramRel=gramRel, language=language)
            for da in data2:
                dar=(json.dumps(da[0], ensure_ascii=False)) #json dump izvornih znakova
                darStrip=dar.strip('"') #moraju se skinuti navodnici da ide u loop
                print("new WordSketch related to", lemmaOdabrana+pos, "in", corpus, ": ", darStrip+pos2)
                sleep(sleepTime)
                graph.run(query, metoda=metoda, userName=userName, apiKey=apiKey, lemmaOdabrana=darStrip, pos=pos2, corpus=corpus, maxitems=maxitems, language=language) #parametri
                sleep(sleepTime)



# =============================================================================
# sketch2neo EuroParl 7  wordSketch harvest
# =============================================================================
def wordSketchEuroParl7(lemmeOdabrane, userName, apiKey, metoda, pos, pos2, corpus, maxitems, language, gramRel):

    for lemmaOdabrana in lemmeOdabrane:
        query="""   
        //trazenje rijeci u gramrel: wsketch? metoda
        WITH ("https://api.sketchengine.eu/bonito/run.cgi/"+
        $metoda+"username="+$userName+"&api_key="+$apiKey+
        "&corpname="+$corpus+"&lemma="+$lemmaOdabrana+"&lpos="+$pos+"&maxitems="+$maxitems+"&format=json") AS url
        CALL apoc.load.json(url) YIELD value
        
        // send request YIELD value.lemma > ako ne dobije podatke sa SketchEngin Corpusa završava upit ELSE ide dalje s procedurom MERGE
        FOREACH (x IN CASE WHEN value.lemma IS NULL THEN [] ELSE [1] END | 
            MERGE (from:Lemma{lempos:value.lemma+value.lpos, language:$language})
            ON CREATE SET from.name=value.lemma
            ON MATCH SET from.lpos=value.lpos
            ON MATCH SET from.corpus_europarl7_en=value.corp_full_name
            ON MATCH SET from.freq_europarl7_en=toInteger(value.freq)
            ON MATCH SET from.relFreq_europarl7_en=toFloat(value.relfreq)
            ON MATCH SET from.primjerLink_europarl7_en='https://api.sketchengine.eu/bonito/run.cgi/view?'+value.lemma_conc_link+';username=dgrguric&api_key=AXEM009NPB4KVF4BEHFZPC28YPR1BR8Y&corpname='+{corpus}
            ON CREATE SET from.language=$language)
        
            //gramatičke relacije pretvorene u nodes
        WITH value as Value
            //nađi u bazi lemu s kojim će se povezivati preko lemposa koji su unique + jezik
        MATCH (lemma:Lemma{lempos:$lemmaOdabrana+$pos, language:$language})
        UNWIND Value.Gramrels as Gramrels
        
            //napravi vezu: lemma ima GrammaticRelations
        FOREACH (
        grels in Gramrels | MERGE (grel:GrammaticRelations{name:Gramrels.name}) ON CREATE SET grel.language=$language
            MERGE (lemma)-[r:HAS_GRAMREL]->(grel)
                ON CREATE SET r.seek_europarl7_en=Gramrels.seek, r.count_europarl7_en=Gramrels.count, r.score_europarl7_en=Gramrels.score
                ON CREATE SET r.corpus_europarl7_en=r.corpus+{corpus}
                ON CREATE SET r.GramRelPrimjerLink_europarl7_en='https://api.sketchengine.eu/bonito/run.cgi/view?q=w'+Gramrels.seek+';username=bperak&api_key=VZSUO4L8NV5Y9WVNWNRFU2JRRXKGTK3E&corpname='+$corpus
        )
        
            //to lemma
        WITH Gramrels AS gramrelSelected
        UNWIND gramrelSelected.Words AS Words
        
            //stvori to lemme iz WSketcha i poveži s lemmafrom
        FOREACH (
        word IN CASE WHEN Words.lempos <> "" THEN [1] ELSE [] END  | 
            MERGE (to:Lemma{lempos:Words.lempos, language:$language}) ON CREATE SET to.name=Words.word
        )
        WITH gramrelSelected, Words
        MATCH (to:Lemma{lempos:Words.lempos, language:$language})
        MATCH(lemmafrom:Lemma{lempos:$lemmaOdabrana+$pos, language:$language})
        WITH lemmafrom, to, gramrelSelected, Words
        
        CALL apoc.merge.relationship(lemmafrom, gramrelSelected.name,{relID:Words.id},{type:gramrelSelected.name, count_europarl7_en:Words.count,score_europarl7_en:Words.score, cm_europarl7_en:Words.cm, seek_europarl7_en:Words.seek, relID:Words.id, exampleLink_europarl7_en:'https://api.sketchengine.eu/bonito/run.cgi/view?q=w'+Words.seek+';username=bperak&api_key=VZSUO4L8NV5Y9WVNWNRFU2JRRXKGTK3E&corpname='+{corpus},language:$language}, to) YIELD rel
        RETURN lemmafrom.name as From, type(rel), to.name as To, rel.count_europarl7_en as Freq, rel.score_europarl7_en as Sq 
        ORDER BY type(rel), Freq, Sq DESC
        """
        graph.run(query, userName=userName, apiKey=apiKey, metoda=metoda, lemmaOdabrana=lemmaOdabrana, pos=pos, corpus=corpus, maxitems=maxitems, language=language) #parametri
    
        for lemmaOdabrana in lemmeOdabrane:
            query2="""
            MATCH (n:Lemma{lempos:$lemmaOdabrana+$pos,language:$language})-[r]-(m:Lemma) 
            WHERE m.lempos ENDS WITH $pos2 AND type(r)=$gramRel AND NOT EXISTS (m.freq_europarl7_en)
            RETURN m.name AS lemmaNova
            """
            data2 = graph.run(query2, userName=userName, apiKey=apiKey, lemmaOdabrana=lemmaOdabrana, pos=pos, pos2=pos2, gramRel=gramRel, language=language)
            
            for da in data2:
                dar=(json.dumps(da[0], ensure_ascii=False)) #json dump izvornih znakova
                darStrip=dar.strip('"') #moraju se skinuti navodnici da ide u loop
                print("new WordSketch related to", lemmaOdabrana+pos, "in", corpus, "gramRel:", gramRel, ": ", darStrip+pos2)                
                sleep(sleepTime)
                graph.run(query, metoda=metoda, userName=userName, apiKey=apiKey, lemmaOdabrana=darStrip, pos=pos, pos2=pos2, corpus=corpus, maxitems=maxitems, language=language) #parametri
                sleep(sleepTime)
    
# =============================================================================
# sketch2neo EuroParl Plenary wordSketch harvest
# =============================================================================
def wordSketchEuroParlPlenary(lemmeOdabrane, userName, apiKey, metoda, pos, pos2, corpus, maxitems, language, gramRel):
    for lemmaOdabrana in lemmeOdabrane:
        query="""
        
    //trazenje rijeci u gramrel: wsketch? metoda
    WITH ("https://api.sketchengine.eu/bonito/run.cgi/"+
    $metoda+"username="+$userName+"&api_key="+$apiKey+
    "&corpname="+$corpus+"&lemma="+$lemmaOdabrana+"&lpos="+$pos+"&maxitems="+$maxitems+"&format=json") AS url
    CALL apoc.load.json(url) YIELD value
    
    // send request YIELD value.lemma > ako ne dobije podatke sa SketchEngin Corpusa završava upit ELSE ide dalje s procedurom MERGE
    FOREACH (x IN CASE WHEN value.lemma IS NULL THEN [] ELSE [1] END | 
    	MERGE (from:Lemma{lempos:value.lemma+value.lpos, language:$language})
    	ON CREATE SET from.name=value.lemma
    	ON MATCH SET from.lpos=value.lpos
    	ON MATCH SET from.corpus=value.corp_full_name
    	ON MATCH SET from.freq_europarl_plenary=toInteger(value.freq)
    	ON MATCH SET from.relFreq_europarl_plenary=toFloat(value.relfreq)
    	ON MATCH SET from.primjerLink_europarl_plenary='https://api.sketchengine.eu/bonito/run.cgi/view?'+value.lemma_conc_link+';username=dgrguric&api_key=AXEM009NPB4KVF4BEHFZPC28YPR1BR8Y&corpname='+{corpus}
    	ON CREATE SET from.language={language})
    
    //gramatičke relacije pretvorene u nodes
    WITH value as Value
    //nađi u bazi lemu s kojim će se povezivati preko lemposa koji su unique + jezik
    MATCH (lemma:Lemma{lempos:$lemmaOdabrana+$pos, language:$language})
    UNWIND Value.Gramrels as Gramrels
    
    //napravi vezu: lemma ima GrammaticRelations
    FOREACH (
    grels in Gramrels | MERGE (grel:GrammaticRelations{name:Gramrels.name}) ON CREATE SET grel.language=$language
        MERGE (lemma)-[r:HAS_GRAMREL]->(grel)
            ON CREATE SET r.seek_europarl_plenary=Gramrels.seek, r.count=Gramrels.count, r.score=Gramrels.score
            ON CREATE SET r.corpus_europarl_plenary=r.corpus+$corpus
            ON CREATE SET r.GramRelPrimjerLink_europarl_plenary='https://api.sketchengine.eu/bonito/run.cgi/view?q=w'+Gramrels.seek+';username=bperak&api_key=VZSUO4L8NV5Y9WVNWNRFU2JRRXKGTK3E&corpname='+{corpus}
    )
    
    	//to lemma
    WITH Gramrels AS gramrelSelected
    UNWIND gramrelSelected.Words AS Words
    
    	//stvori to lemme iz WSketcha i poveži s lemmafrom
    FOREACH (
    word IN CASE WHEN Words.lempos <> "" THEN [1] ELSE [] END  | 
        MERGE (to:Lemma{lempos:Words.lempos, language:$language}) ON CREATE SET to.name=Words.word
    )
    WITH gramrelSelected, Words
    MATCH (to:Lemma{lempos:Words.lempos, language:$language})
    MATCH(lemmafrom:Lemma{lempos:$lemmaOdabrana+$pos, language:$language})
    WITH lemmafrom, to, gramrelSelected, Words
    
    CALL apoc.merge.relationship(lemmafrom, gramrelSelected.name,{relID:Words.id},{type:gramrelSelected.name, count_europarl_plenary:Words.count,score_europarl_plenary:Words.score, cm_europarl_plenary:Words.cm, seek_europarl_plenary:Words.seek, relID:Words.id, exampleLink_europarl_plenary:'https://api.sketchengine.eu/bonito/run.cgi/view?q=w'+Words.seek+';username=bperak&api_key=VZSUO4L8NV5Y9WVNWNRFU2JRRXKGTK3E&corpname='+{corpus},language:$language}, to) YIELD rel
    RETURN lemmafrom.name as From, type(rel), to.name as To, rel.count_europarl_plenary as Freq, rel.score_europarl_plenary as Sq 
    ORDER BY type(rel), Freq, Sq DESC
    """
        graph.run(query, userName=userName, apiKey=apiKey, metoda=metoda, lemmaOdabrana=lemmaOdabrana, pos=pos, corpus=corpus, maxitems=maxitems, language=language) #parametri

        for lemmaOdabrana in lemmeOdabrane:
            query2="""
            MATCH (n:Lemma{lempos:$lemmaOdabrana+$pos,language:$language})-[r]-(m:Lemma) 
            WHERE m.lempos ENDS WITH $pos2 AND type(r)=$gramRel AND NOT EXISTS (m.freq_europarl_plenary)
            RETURN m.name AS lemmaNova
            """
            data2 = graph.run(query2, userName=userName, apiKey=apiKey, lemmaOdabrana=lemmaOdabrana, pos=pos, pos2=pos2, gramRel=gramRel, language=language)
            for da in data2:
                dar=(json.dumps(da[0], ensure_ascii=False)) #json dump izvornih znakova
                darStrip=dar.strip('"') #moraju se skinuti navodnici da ide u loop         
                print("new WordSketch related to", lemmaOdabrana+pos, "in", corpus, "gramRel:", gramRel, ": ", darStrip+pos2)
                sleep(sleepTime)
                graph.run(query, metoda=metoda, userName=userName, apiKey=apiKey, lemmaOdabrana=darStrip, pos=pos, pos2=pos2, corpus=corpus, maxitems=maxitems, language=language) #parametri
                sleep(sleepTime)
                
                
# =============================================================================
# sketch2neo hrwac22 wordSketch harvest
# =============================================================================
def wordSketchHrWac22(lemmeOdabrane, userName, apiKey, metoda, pos, pos2, corpus, maxitems, language, gramRel):
    for lemmaOdabrana in lemmeOdabrane:
        query="""       
    //trazenje rijeci u gramrel: wsketch? metoda
    WITH ("https://api.sketchengine.eu/bonito/run.cgi/"+
    $metoda+"username="+$userName+"&api_key="+$apiKey+
    "&corpname="+$corpus+"&lemma="+$lemmaOdabrana+"&lpos="+$pos+"&maxitems="+$maxitems+"&expand_seppage=1&format=json") AS url
    CALL apoc.load.json(url) YIELD value
    
    // send request YIELD value.lemma > ako ne dobije podatke sa SketchEngin Corpusa završava upit ELSE ide dalje s procedurom MERGE
    FOREACH (x IN CASE WHEN value.lemma IS NULL THEN [] ELSE [1] END | 
    	MERGE (from:Lemma{lempos:value.lemma+value.lpos, language:$language})
    	ON CREATE SET from.name=value.lemma
    	ON MATCH SET from.lpos=value.lpos
    	ON MATCH SET from.corpus=value.corp_full_name
    	ON MATCH SET from.freq_hrwac22=toInteger(value.freq)
    	ON MATCH SET from.relFreq_hrwac22=toFloat(value.relfreq)
    	ON MATCH SET from.primjerLink_hrwac22='https://api.sketchengine.eu/bonito/run.cgi/view?'+value.lemma_conc_link+';username='+$userName+'&api_key='+$apiKey+'&corpname='+$corpus //đđ ovo je dobar link sa parametrima
    	ON CREATE SET from.language=$language)
    
    	//gramatičke relacije pretvorene u nodes
    WITH value as Value
        //nađi u bazi lemu s kojim će se povezivati preko lemposa koji su unique + jezik
    MATCH (lemma:Lemma{lempos:$lemmaOdabrana+$pos, language:$language})
    UNWIND Value.Gramrels as Gramrels
    
    	//napravi vezu: lemma ima GrammaticRelations
    FOREACH (
    grels in Gramrels | MERGE (grel:GrammaticRelations{name:Gramrels.name}) ON CREATE SET grel.language=$language
        MERGE (lemma)-[r:HAS_GRAMREL]->(grel)
            ON CREATE SET r.seek_hrwac22=Gramrels.seek, r.count=Gramrels.count, r.score=Gramrels.score
            ON CREATE SET r.corpus_hrwac22=r.corpus+$corpus
            ON CREATE SET r.GramRelPrimjerLink_hrwac22='https://api.sketchengine.eu/bonito/run.cgi/view?q=w'+Gramrels.seek+';username=bperak&api_key=VZSUO4L8NV5Y9WVNWNRFU2JRRXKGTK3E&corpname='+{corpus}
    )
    
    	//to lemma
    WITH Gramrels AS gramrelSelected
    UNWIND gramrelSelected.Words AS Words
    
    	//stvori to lemme iz WSketcha i poveži s lemmafrom
    FOREACH (
    word IN CASE WHEN Words.lempos <> "" THEN [1] ELSE [] END  | 
        MERGE (to:Lemma{lempos:Words.lempos, language:$language}) ON CREATE SET to.name=Words.word
    )
    WITH gramrelSelected, Words
    MATCH (to:Lemma{lempos:Words.lempos, language:$language})
    MATCH(lemmafrom:Lemma{lempos:$lemmaOdabrana+$pos, language:$language})
    WITH lemmafrom, to, gramrelSelected, Words
    
    CALL apoc.merge.relationship(lemmafrom, gramrelSelected.name,{relID:Words.id},{type:gramrelSelected.name, count_hrwac22:Words.count,score_hrwac22:Words.score, cm_hrwac22:Words.cm, seek_hrwac22:Words.seek, relID:Words.id, exampleLink_hrwac22:'https://api.sketchengine.eu/bonito/run.cgi/view?q=w'+Words.seek+';username=bperak&api_key=VZSUO4L8NV5Y9WVNWNRFU2JRRXKGTK3E&corpname='+{corpus},language:$language}, to) YIELD rel
    RETURN lemmafrom.name as From, type(rel), to.name as To, rel.count_hrwac22 as Freq, rel.score_hrwac22 as Sq 
    ORDER BY type(rel), Freq, Sq DESC
    """
        graph.run(query, userName=userName, apiKey=apiKey, metoda=metoda, lemmaOdabrana=lemmaOdabrana, pos=pos, corpus=corpus, maxitems=maxitems, language=language) #parametri
    
        for lemmaOdabrana in lemmeOdabrane:
            query2="""
            MATCH (n:Lemma{lempos:$lemmaOdabrana+$pos,language:$language})-[r]-(m:Lemma) 
            WHERE m.lempos ENDS WITH $pos2 AND type(r)=$gramRel AND NOT EXISTS (m.freq_hrwac22)
            RETURN m.name AS lemmaNova
            """
            data2 = graph.run(query2, userName=userName, apiKey=apiKey, lemmaOdabrana=lemmaOdabrana, pos=pos, pos2=pos2, gramRel=gramRel, language=language)
            for da in data2:
                dar=(json.dumps(da[0], ensure_ascii=False)) #json dump izvornih znakova
                darStrip=dar.strip('"') #moraju se skinuti navodnici da ide u loop
                print("new WordSketch related to", lemmaOdabrana+pos, "in", corpus, "gramRel:", gramRel, ": ", darStrip+pos2)
                sleep(sleepTime)
                graph.run(query, metoda=metoda, userName=userName, apiKey=apiKey, lemmaOdabrana=darStrip, pos=pos2, corpus=corpus, maxitems=maxitems, language=language) #parametri
                sleep(sleepTime)


# =============================================================================
# itTenTen 16
# =============================================================================

def wordSketchIttenten16(lemmeOdabrane, userName, apiKey, metoda, pos, pos2, corpus, maxitems, language, gramRel):
    for lemmaOdabrana in lemmeOdabrane:
        query="""       
    //trazenje rijeci u gramrel: wsketch? metoda
    WITH ("https://api.sketchengine.eu/bonito/run.cgi/"+
    $metoda+"username="+$userName+"&api_key="+$apiKey+
    "&corpname="+$corpus+"&lemma="+$lemmaOdabrana+"&lpos="+$pos+"&maxitems="+$maxitems+"&expand_seppage=1&format=json") AS url
    CALL apoc.load.json(url) YIELD value
    
    // send request YIELD value.lemma > ako ne dobije podatke sa SketchEngin Corpusa završava upit ELSE ide dalje s procedurom MERGE
    FOREACH (x IN CASE WHEN value.lemma IS NULL THEN [] ELSE [1] END | 
    	MERGE (from:Lemma{lempos:value.lemma+value.lpos, language:$language})
    	ON CREATE SET from.name=value.lemma
    	ON MATCH SET from.lpos=value.lpos
    	ON MATCH SET from.corpus=value.corp_full_name
    	ON MATCH SET from.freq_ittenten16_2=toInteger(value.freq)
    	ON MATCH SET from.relFreq_ittenten16_2=toFloat(value.relfreq)
    	ON MATCH SET from.primjerLink_ittenten16_2='https://api.sketchengine.eu/bonito/run.cgi/view?'+value.lemma_conc_link+';username='+$userName+'&api_key='+$apiKey+'&corpname='+$corpus //đđ ovo je dobar link sa parametrima
    	ON CREATE SET from.language=$language)
    
    	//gramatičke relacije pretvorene u nodes
    WITH value as Value
        //nađi u bazi lemu s kojim će se povezivati preko lemposa koji su unique + jezik
    MATCH (lemma:Lemma{lempos:$lemmaOdabrana+$pos, language:$language})
    UNWIND Value.Gramrels as Gramrels
    
    	//napravi vezu: lemma ima GrammaticRelations
    FOREACH (
    grels in Gramrels | MERGE (grel:GrammaticRelations{name:Gramrels.name}) ON CREATE SET grel.language=$language
        MERGE (lemma)-[r:HAS_GRAMREL]->(grel)
            ON CREATE SET r.seek_ittenten16_2=Gramrels.seek, r.count=Gramrels.count, r.score=Gramrels.score
            ON CREATE SET r.corpus_ittenten16_2=r.corpus+$corpus
            ON CREATE SET r.GramRelPrimjerLink_ittenten16_2='https://api.sketchengine.eu/bonito/run.cgi/view?q=w'+Gramrels.seek+';username=bperak&api_key=VZSUO4L8NV5Y9WVNWNRFU2JRRXKGTK3E&corpname='+{corpus}
    )
    
    	//to lemma
    WITH Gramrels AS gramrelSelected
    UNWIND gramrelSelected.Words AS Words
    
    	//stvori to lemme iz WSketcha i poveži s lemmafrom
    FOREACH (
    word IN CASE WHEN Words.lempos <> "" THEN [1] ELSE [] END  | 
        MERGE (to:Lemma{lempos:Words.lempos, language:$language}) ON CREATE SET to.name=Words.word
    )
    WITH gramrelSelected, Words
    MATCH (to:Lemma{lempos:Words.lempos, language:$language})
    MATCH(lemmafrom:Lemma{lempos:$lemmaOdabrana+$pos, language:$language})
    WITH lemmafrom, to, gramrelSelected, Words
    
    CALL apoc.merge.relationship(lemmafrom, gramrelSelected.name,{relID:Words.id},{type:gramrelSelected.name, count_ittenten16_2:Words.count,score_ittenten16_2:Words.score, cm_ittenten16_2:Words.cm, seek_ittenten16_2:Words.seek, relID:Words.id, exampleLink_ittenten16_2:'https://api.sketchengine.eu/bonito/run.cgi/view?q=w'+Words.seek+';username=bperak&api_key=VZSUO4L8NV5Y9WVNWNRFU2JRRXKGTK3E&corpname='+{corpus},language:$language}, to) YIELD rel
    RETURN lemmafrom.name as From, type(rel), to.name as To, rel.count_ittenten16_2 as Freq, rel.score_ittenten16_2 as Sq 
    ORDER BY type(rel), Freq, Sq DESC
    """
        graph.run(query, userName=userName, apiKey=apiKey, metoda=metoda, lemmaOdabrana=lemmaOdabrana, pos=pos, corpus=corpus, maxitems=maxitems, language=language) #parametri
    
        for lemmaOdabrana in lemmeOdabrane:
            query2="""
            MATCH (n:Lemma{lempos:$lemmaOdabrana+$pos,language:$language})-[r]-(m:Lemma) 
            WHERE m.lempos ENDS WITH $pos2 AND type(r)=$gramRel AND NOT EXISTS (m.freq_ittenten16_2)
            RETURN m.name AS lemmaNova
            """
            data2 = graph.run(query2, userName=userName, apiKey=apiKey, lemmaOdabrana=lemmaOdabrana, pos=pos, pos2=pos2, gramRel=gramRel, language=language)
            for da in data2:
                dar=(json.dumps(da[0], ensure_ascii=False)) #json dump izvornih znakova
                darStrip=dar.strip('"') #moraju se skinuti navodnici da ide u loop
                print("new WordSketch related to", lemmaOdabrana+pos, "in", corpus, "gramRel:", gramRel, ": ", darStrip+pos2)
                sleep(sleepTime)
                graph.run(query, metoda=metoda, userName=userName, apiKey=apiKey, lemmaOdabrana=darStrip, pos=pos2, corpus=corpus, maxitems=maxitems, language=language) #parametri
                sleep(sleepTime)

# =============================================================================
# sketch2neo eng_jsi_newsfeed_virt wordSketch harvest
# =============================================================================
def wordSketchEng_jsi_newsfeed_virt(lemmeOdabrane, userName, apiKey, metoda, pos, pos2, corpus, maxitems, language, gramRel):
    for lemmaOdabrana in lemmeOdabrane:
        query="""       
        //trazenje rijeci u gramrel: wsketch? metoda
        WITH ("https://api.sketchengine.eu/bonito/run.cgi/"+
        $metoda+"username="+$userName+"&api_key="+$apiKey+
        "&corpname="+$corpus+"&lemma="+$lemmaOdabrana+"&lpos="+$pos+"&maxitems="+$maxitems+"&format=json") AS url
        CALL apoc.load.json(url) YIELD value
        
        // send request YIELD value.lemma > ako ne dobije podatke sa SketchEngin Corpusa završava upit ELSE ide dalje s procedurom MERGE
        FOREACH (x IN CASE WHEN value.lemma IS NULL THEN [] ELSE [1] END | 
            MERGE (from:Lemma{lempos:value.lemma+value.lpos, language:$language})
            ON CREATE SET from.name=value.lemma
            ON MATCH SET from.lpos=value.lpos
            ON MATCH SET from.corpus=value.corp_full_name
            ON MATCH SET from.freq_eng_jsi_newsfeed_virt=toInteger(value.freq)
            ON MATCH SET from.relFreq_eng_jsi_newsfeed_virt=toFloat(value.relfreq)
            ON MATCH SET from.primjerLink_eng_jsi_newsfeed_virt='https://api.sketchengine.eu/bonito/run.cgi/view?'+value.lemma_conc_link+';username='+$userName+'&api_key='+$apiKey+'&corpname='+$corpus //đđ ovo je dobar link sa parametrima
            ON CREATE SET from.language=$language)
        
            //gramatičke relacije pretvorene u nodes
        WITH value as Value
            //nađi u bazi lemu s kojim će se povezivati preko lemposa koji su unique + jezik
        MATCH (lemma:Lemma{lempos:$lemmaOdabrana+$pos, language:$language})
        UNWIND Value.Gramrels as Gramrels
        
            //napravi vezu: lemma ima GrammaticRelations
        FOREACH (
        grels in Gramrels | MERGE (grel:GrammaticRelations{name:Gramrels.name}) ON CREATE SET grel.language=$language
            MERGE (lemma)-[r:HAS_GRAMREL]->(grel)
                ON CREATE SET r.seek_eng_jsi_newsfeed_virt=Gramrels.seek, r.count=Gramrels.count, r.score=Gramrels.score
                ON CREATE SET r.corpus_eng_jsi_newsfeed_virt=r.corpus+$corpus
                ON CREATE SET r.GramRelPrimjerLink_eng_jsi_newsfeed_virt='https://api.sketchengine.eu/bonito/run.cgi/view?q=w'+Gramrels.seek+';username='+$userName+'&api_key='+$apiKey+'&corpname='+$corpus
        )
        
            //to lemma
        WITH Gramrels AS gramrelSelected
        UNWIND gramrelSelected.Words AS Words
        
            //stvori to lemme iz WSketcha i poveži s lemmafrom
        FOREACH (
        word IN CASE WHEN Words.lempos <> "" THEN [1] ELSE [] END  | 
            MERGE (to:Lemma{lempos:Words.lempos, language:$language}) ON CREATE SET to.name=Words.word
        )
        WITH gramrelSelected, Words
        MATCH (to:Lemma{lempos:Words.lempos, language:$language})
        MATCH(lemmafrom:Lemma{lempos:$lemmaOdabrana+$pos, language:$language})
        WITH lemmafrom, to, gramrelSelected, Words
        
        CALL apoc.merge.relationship(lemmafrom, gramrelSelected.name,{relID:Words.id},{type:gramrelSelected.name, count_eng_jsi_newsfeed_virt:Words.count,score_eng_jsi_newsfeed_virt:Words.score, cm_eng_jsi_newsfeed_virt:Words.cm, seek_eng_jsi_newsfeed_virt:Words.seek, relID:Words.id, exampleLink_eng_jsi_newsfeed_virt:'https://api.sketchengine.eu/bonito/run.cgi/view?q=w'+Words.seek+';username='+$userName+'&api_key='+$apiKey+'&corpname='+$corpus,language:$language}, to) YIELD rel
        RETURN lemmafrom.name as From, type(rel), to.name as To, rel.count_eng_jsi_newsfeed_virt as Freq, rel.score_eng_jsi_newsfeed_virt as Sq 
        ORDER BY type(rel), Freq, Sq DESC
        """
        graph.run(query, userName=userName, apiKey=apiKey, metoda=metoda, lemmaOdabrana=lemmaOdabrana, pos=pos, corpus=corpus, maxitems=maxitems, language=language) #parametri
    
        for lemmaOdabrana in lemmeOdabrane:
            query2="""
            MATCH (n:Lemma{lempos:$lemmaOdabrana+$pos,language:$language})-[r]-(m:Lemma) 
            WHERE m.lempos ENDS WITH $pos2 AND type(r)=$gramRel AND NOT EXISTS (m.freq_eng_jsi_newsfeed_virt)
            RETURN m.name AS lemmaNova
            """
            data2 = graph.run(query2, userName=userName, apiKey=apiKey, lemmaOdabrana=lemmaOdabrana, pos=pos, pos2=pos2, gramRel=gramRel, language=language)
            for da in data2:
                dar=(json.dumps(da[0], ensure_ascii=False)) #json dump izvornih znakova
                darStrip=dar.strip('"') #moraju se skinuti navodnici da ide u loop
                print("new WordSketch related to", lemmaOdabrana+pos, "in", corpus, ": ", darStrip+pos2)
                sleep(sleepTime)
                graph.run(query, metoda=metoda, userName=userName, apiKey=apiKey, lemmaOdabrana=darStrip, pos=pos2, corpus=corpus, maxitems=maxitems, language=language) #parametri
                sleep(sleepTime)


# =============================================================================
# sketch2neo eng_jsi_newsfeed_lastmonth wordSketch harvest
# =============================================================================
def wordSketchEng_jsi_newsfeed_lastmonth(lemmeOdabrane, userName, apiKey, metoda, pos, pos2, corpus, maxitems, language, gramRel):
    for lemmaOdabrana in lemmeOdabrane:
        query="""       
        //trazenje rijeci u gramrel: wsketch? metoda
        WITH ("https://api.sketchengine.eu/bonito/run.cgi/"+
        $metoda+"username="+$userName+"&api_key="+$apiKey+
        "&corpname="+$corpus+"&lemma="+$lemmaOdabrana+"&lpos="+$pos+"&maxitems="+$maxitems+"&format=json") AS url
        CALL apoc.load.json(url) YIELD value
        
        // send request YIELD value.lemma > ako ne dobije podatke sa SketchEngin Corpusa završava upit ELSE ide dalje s procedurom MERGE
        FOREACH (x IN CASE WHEN value.lemma IS NULL THEN [] ELSE [1] END | 
            MERGE (from:Lemma{lempos:value.lemma+value.lpos, language:$language})
            ON CREATE SET from.name=value.lemma
            ON MATCH SET from.lpos=value.lpos
            ON MATCH SET from.corpus=value.corp_full_name
            ON MATCH SET from.freq_eng_jsi_newsfeed_lastmonth=toInteger(value.freq)
            ON MATCH SET from.relFreq_eng_jsi_newsfeed_lastmonth=toFloat(value.relfreq)
            ON MATCH SET from.primjerLink_eng_jsi_newsfeed_lastmonth='https://api.sketchengine.eu/bonito/run.cgi/view?'+value.lemma_conc_link+';username='+$userName+'&api_key='+$apiKey+'&corpname='+$corpus //đđ ovo je dobar link sa parametrima
            ON CREATE SET from.language=$language)
        
            //gramatičke relacije pretvorene u nodes
        WITH value as Value
            //nađi u bazi lemu s kojim će se povezivati preko lemposa koji su unique + jezik
        MATCH (lemma:Lemma{lempos:$lemmaOdabrana+$pos, language:$language})
        UNWIND Value.Gramrels as Gramrels
        
            //napravi vezu: lemma ima GrammaticRelations
        FOREACH (
        grels in Gramrels | MERGE (grel:GrammaticRelations{name:Gramrels.name}) ON CREATE SET grel.language=$language
            MERGE (lemma)-[r:HAS_GRAMREL]->(grel)
                ON CREATE SET r.seek_eng_jsi_newsfeed_lastmonth=Gramrels.seek, r.count=Gramrels.count, r.score=Gramrels.score
                ON CREATE SET r.corpus_eng_jsi_newsfeed_lastmonth=r.corpus+$corpus
                ON CREATE SET r.GramRelPrimjerLink_eng_jsi_newsfeed_lastmonth='https://api.sketchengine.eu/bonito/run.cgi/view?q=w'+Gramrels.seek+';username='+$userName+'&api_key='+$apiKey+'&corpname='+$corpus
        )
        
            //to lemma
        WITH Gramrels AS gramrelSelected
        UNWIND gramrelSelected.Words AS Words
        
            //stvori to lemme iz WSketcha i poveži s lemmafrom
        FOREACH (
        word IN CASE WHEN Words.lempos <> "" THEN [1] ELSE [] END  | 
            MERGE (to:Lemma{lempos:Words.lempos, language:$language}) ON CREATE SET to.name=Words.word
        )
        WITH gramrelSelected, Words
        MATCH (to:Lemma{lempos:Words.lempos, language:$language})
        MATCH(lemmafrom:Lemma{lempos:$lemmaOdabrana+$pos, language:$language})
        WITH lemmafrom, to, gramrelSelected, Words
        
        CALL apoc.merge.relationship(lemmafrom, gramrelSelected.name,{relID:Words.id},{type:gramrelSelected.name, count_eng_jsi_newsfeed_lastmonth:Words.count,score_eng_jsi_newsfeed_lastmonth:Words.score, cm_eng_jsi_newsfeed_lastmonth:Words.cm, seek_eng_jsi_newsfeed_lastmonth:Words.seek, relID:Words.id, exampleLink_eng_jsi_newsfeed_lastmonth:'https://api.sketchengine.eu/bonito/run.cgi/view?q=w'+Words.seek+';username='+$userName+'&api_key='+$apiKey+'&corpname='+$corpus,language:$language}, to) YIELD rel
        RETURN lemmafrom.name as From, type(rel), to.name as To, rel.count_eng_jsi_newsfeed_lastmonth as Freq, rel.score_eng_jsi_newsfeed_lastmonth as Sq 
        ORDER BY type(rel), Freq, Sq DESC
        """
        graph.run(query, userName=userName, apiKey=apiKey, metoda=metoda, lemmaOdabrana=lemmaOdabrana, pos=pos, corpus=corpus, maxitems=maxitems, language=language) #parametri
    
        for lemmaOdabrana in lemmeOdabrane:
            query2="""
            MATCH (n:Lemma{lempos:$lemmaOdabrana+$pos,language:$language})-[r]-(m:Lemma) 
            WHERE m.lempos ENDS WITH $pos2 AND type(r)=$gramRel AND NOT EXISTS (m.freq_eng_jsi_newsfeed_lastmonth)
            RETURN m.name AS lemmaNova
            """
            data2 = graph.run(query2, userName=userName, apiKey=apiKey, lemmaOdabrana=lemmaOdabrana, pos=pos, pos2=pos2, gramRel=gramRel, language=language)
            for da in data2:
                dar=(json.dumps(da[0], ensure_ascii=False)) #json dump izvornih znakova
                darStrip=dar.strip('"') #moraju se skinuti navodnici da ide u loop
                print("new WordSketch related to", lemmaOdabrana+pos, "in", corpus, ": ", darStrip+pos2)
                sleep(sleepTime)
                graph.run(query, metoda=metoda, userName=userName, apiKey=apiKey, lemmaOdabrana=darStrip, pos=pos2, corpus=corpus, maxitems=maxitems, language=language) #parametri
                sleep(sleepTime)
    

# =============================================================================
# sketch2neo deu_jsi_newsfeed_virt wordSketch harvest
# =============================================================================
def wordSketchDeu_jsi_newsfeed_virt(lemmeOdabrane, userName, apiKey, metoda, pos, pos2, corpus, maxitems, language, gramRel):
    for lemmaOdabrana in lemmeOdabrane:
        query="""       
        //trazenje rijeci u gramrel: wsketch? metoda
        WITH ("https://api.sketchengine.eu/bonito/run.cgi/"+
        $metoda+"username="+$userName+"&api_key="+$apiKey+
        "&corpname="+$corpus+"&lemma="+$lemmaOdabrana+"&lpos="+$pos+"&maxitems="+$maxitems+"&format=json") AS url
        CALL apoc.load.json(url) YIELD value
        
        // send request YIELD value.lemma > ako ne dobije podatke sa SketchEngin Corpusa završava upit ELSE ide dalje s procedurom MERGE
        FOREACH (x IN CASE WHEN value.lemma IS NULL THEN [] ELSE [1] END | 
            MERGE (from:Lemma{lempos:value.lemma+value.lpos, language:$language})
            ON CREATE SET from.name=value.lemma
            ON MATCH SET from.lpos=value.lpos
            ON MATCH SET from.corpus_deu_jsi_newsfeed_virt=value.corp_full_name
            ON MATCH SET from.freq_deu_jsi_newsfeed_virt=toInteger(value.freq)
            ON MATCH SET from.relFreq_deu_jsi_newsfeed_virt=toFloat(value.relfreq)
            ON MATCH SET from.primjerLink_deu_jsi_newsfeed_virt='https://api.sketchengine.eu/bonito/run.cgi/view?'+value.lemma_conc_link+';username='+$userName+'&api_key='+$apiKey+'&corpname='+$corpus //đđ ovo je dobar link sa parametrima
            ON CREATE SET from.language=$language)
        
            //gramatičke relacije pretvorene u nodes
        WITH value as Value
            //nađi u bazi lemu s kojim će se povezivati preko lemposa koji su unique + jezik
        MATCH (lemma:Lemma{lempos:$lemmaOdabrana+$pos, language:$language})
        UNWIND Value.Gramrels as Gramrels
        
            //napravi vezu: lemma ima GrammaticRelations
        FOREACH (
        grels in Gramrels | MERGE (grel:GrammaticRelations{name:Gramrels.name}) ON CREATE SET grel.language=$language
            MERGE (lemma)-[r:HAS_GRAMREL]->(grel)
                ON CREATE SET r.seek_deu_jsi_newsfeed_virt=Gramrels.seek, r.count=Gramrels.count, r.score=Gramrels.score
                ON CREATE SET r.corpus_deu_jsi_newsfeed_virt=r.corpus+$corpus
                ON CREATE SET r.GramRelPrimjerLink_deu_jsi_newsfeed_virt='https://api.sketchengine.eu/bonito/run.cgi/view?q=w'+Gramrels.seek+';username='+$userName+'&api_key='+$apiKey+'&corpname='+$corpus
        )
        
            //to lemma
        WITH Gramrels AS gramrelSelected
        UNWIND gramrelSelected.Words AS Words
        
            //stvori to lemme iz WSketcha i poveži s lemmafrom
        FOREACH (
        word IN CASE WHEN Words.lempos <> "" THEN [1] ELSE [] END  | 
            MERGE (to:Lemma{lempos:Words.lempos, language:$language}) ON CREATE SET to.name=Words.word
        )
        WITH gramrelSelected, Words
        MATCH (to:Lemma{lempos:Words.lempos, language:$language})
        MATCH(lemmafrom:Lemma{lempos:$lemmaOdabrana+$pos, language:$language})
        WITH lemmafrom, to, gramrelSelected, Words
        
        CALL apoc.merge.relationship(lemmafrom, gramrelSelected.name,{relID:Words.id},{type:gramrelSelected.name, count_deu_jsi_newsfeed_virt:Words.count,score_deu_jsi_newsfeed_virt:Words.score, cm_deu_jsi_newsfeed_virt:Words.cm, seek_deu_jsi_newsfeed_virt:Words.seek, relID:Words.id, exampleLink_deu_jsi_newsfeed_virt:'https://api.sketchengine.eu/bonito/run.cgi/view?q=w'+Words.seek+';username='+$userName+'&api_key='+$apiKey+'&corpname='+$corpus,language:$language}, to) YIELD rel
        RETURN lemmafrom.name as From, type(rel), to.name as To, rel.count_deu_jsi_newsfeed_virt as Freq, rel.score_deu_jsi_newsfeed_virt as Sq 
        ORDER BY type(rel), Freq, Sq DESC
        """
        graph.run(query, userName=userName, apiKey=apiKey, metoda=metoda, lemmaOdabrana=lemmaOdabrana, pos=pos, corpus=corpus, maxitems=maxitems, language=language) #parametri
    
        for lemmaOdabrana in lemmeOdabrane:
            query2="""
            MATCH (n:Lemma{lempos:$lemmaOdabrana+$pos,language:$language})-[r]-(m:Lemma) 
            WHERE m.lempos ENDS WITH $pos2 AND type(r)=$gramRel AND NOT EXISTS (m.freq_deu_jsi_newsfeed_virt)
            RETURN m.name AS lemmaNova
            """
            data2 = graph.run(query2, userName=userName, apiKey=apiKey, lemmaOdabrana=lemmaOdabrana, pos=pos, pos2=pos2, gramRel=gramRel, language=language)
            for da in data2:
                dar=(json.dumps(da[0], ensure_ascii=False)) #json dump izvornih znakova
                darStrip=dar.strip('"') #moraju se skinuti navodnici da ide u loop
                print("new WordSketch related to", lemmaOdabrana+pos, "in", corpus, ": ", darStrip+pos2)
                sleep(sleepTime)
                graph.run(query, metoda=metoda, userName=userName, apiKey=apiKey, lemmaOdabrana=darStrip, pos=pos2, corpus=corpus, maxitems=maxitems, language=language) #parametri
                sleep(sleepTime)
