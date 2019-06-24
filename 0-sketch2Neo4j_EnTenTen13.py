# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:00:48 2019

@author: Benedikt Perak bperak@uniri.hr, CulturalComplexityLab University of Rijeka
"""
from py2neo import Graph
import json
from time import sleep
import getpass

# =============================================================================
# Connecting to the Neo4j Database
# =============================================================================
print("Please, activate your local version of the Neo4j graph database")
def neo4jconnect():
    
    proceed=input("Type y when you have Neo4j up and running! ")
    if proceed =="y":
        print("We will now connect to the Neo4j graph database. Please, input your  ")
    else:
        print("Please spin up your Neo4j graph database! https://neo4j.com/")
        
    # Set up authentication parameters
    graphUser = input('Neo4j login: ')
    graphPass = getpass.getpass('Neo4j password:')
    return (graphUser, graphPass)

author=neo4jconnect() 
#author=[user, pass]
graph = Graph("bolt://localhost:7687", auth=(author[0], author[1]))
print ("Connected to dedicated Neo4j graph database")
   
    
# =============================================================================
# Setting up indexes and constraints the Neo4j database
# =============================================================================
def createIndexes():
    if n=='y':
        query1="""
        CREATE INDEX ON :Lemma(name)
        """
        graph.run(query1)
        
        query2="""
        CREATE INDEX ON :Lemma(lempos)
        """
        graph.run(query2)
        
        query3="""
        CREATE INDEX ON :Lemma(language)
        """
        graph.run(query3)
        
        print("Indexes created")
    
n=input('Do you want to create Indexes in Neo4j (y/n) ') 
#n='n'
if n=='y':
    createIndexes()
else:
    pass

# =============================================================================
# Connect to SketchEngine
# =============================================================================
# Set authentication as described by: https://www.sketchengine.eu/documentation/api-documentation/
# User details: https://app.sketchengine.eu/#my

#
print("Setting up the web connection to SketchEngine API. \nIf needed, check your API credentials on the https://app.sketchengine.eu/#my ")
userName=input('SketchEngine API Username: ' )
#userName=''
apiKey= input('Sketch Engine API key: ')
#apiKey=''

# Testing option
#userName= 'api_testing'
#apiKey= 'YNSC0B9OXN57XB48T9HWUFFLPY4TZ6OE'


# =============================================================================
# English ententen13_tt2_1 script
# =============================================================================
print ("This script will harvest Word Sketches for the EnTenTen13 corpus https://app.sketchengine.eu/#dashboard?corpname=preloaded%2Fententen13_tt2_1")

print("Set the search parameters")
# =============================================================================
# parameters
# =============================================================================
#corpus = "preloaded/europarl7_en"
corpus = "preloaded/ententen13_tt2_1" #name of the corpus in the Sketch Engin
# ! to create different corpus change values in the lines containing lemma  //from,  //to

#lemma
les=input("Please, type the WordSketch lemma! \nYou can choose multiple lemmas of a same POS, separated with commma n\Ex: love,hate,husband, etc.: \n").split(",")
lemmeOdabrane=[x.strip() for x in les]

#pos
pos = "-"+str(input("Choose your part of speech type: n (for nouns)|j (for adjectives)|v (for verbs): "))

#maxitems
maxitems= input("Set max wordSketch search for a source lexeme. Ex.: 50 \n").strip() 

#default parameters
metoda = "wsketch?" # Method of extraction (Wordsketch: wsketch?, Thesaurus: )

#and/or default 
#gramatickaRelacija=["e_o"] #Gramatičke relacija - može se odabrati više, npr. ["koordinacija", "subjekt_od"] 
 
language='en' #language hr, en, it, ...
print("All parameters set. Starting downloading the WordSketches for lemmas: ", list(lemmeOdabrane), pos, "maxItems=",maxitems, "method=", metoda)


# =============================================================================
# MAIN SCRIPT
# =============================================================================
def wordSketchEnTenTen13(lemmeOdabrane):
    for lemmaOdabrana in lemmeOdabrane:
        query="""
        
    //trazenje rijeci u gramrel: wsketch? metoda
    WITH ("https://api.sketchengine.co.uk/bonito/run.cgi/"+
    $metoda+"username="+$userName+"&api_key="+$apiKey+
    "&corpname="+$corpus+"&lemma="+$lemmaOdabrana+"&lpos="+$pos+"&maxitems="+$maxitems+"&format=json") AS url
    CALL apoc.load.json(url) YIELD value
    
    // send request YIELD value.lemma > ako ne dobije podatke sa SketchEngin Corpusa završava upit ELSE ide dalje s procedurom MERGE
    FOREACH (x IN CASE WHEN value.lemma IS NULL THEN [] ELSE [1] END | 
    	MERGE (from:Lemma{lempos:value.lemma+value.lpos, language:{language}})
    	ON CREATE SET from.name=value.lemma
    	ON MATCH SET from.lpos=value.lpos
    	ON MATCH SET from.corpus=value.corp_full_name
    	ON MATCH SET from.freq_ententen13_tt2_1=toInteger(value.freq)
    	ON MATCH SET from.relFreq_ententen13_tt2_1=toFloat(value.relfreq)
    	ON MATCH SET from.primjerLink_ententen13_tt2_1='https://api.sketchengine.co.uk/bonito/run.cgi/view?'+value.lemma_conc_link+';username=dgrguric&api_key=AXEM009NPB4KVF4BEHFZPC28YPR1BR8Y&corpname='+{corpus}
    	ON CREATE SET from.language={language})
    
    	//gramatičke relacije pretvorene u nodes
    WITH value as Value
        //nađi u bazi lemu s kojim će se povezivati preko lemposa koji su unique + jezik
    MATCH (lemma:Lemma{lempos:{lemmaOdabrana}+{pos}, language:{language}})
    UNWIND Value.Gramrels as Gramrels
    
    	//napravi vezu: lemma ima GrammaticRelations
    FOREACH (
    grels in Gramrels | MERGE (grel:GrammaticRelations{name:Gramrels.name}) ON CREATE SET grel.language=$language
        MERGE (lemma)-[r:HAS_GRAMREL]->(grel)
            ON CREATE SET r.seek_ententen13_tt2_1=Gramrels.seek, r.count=Gramrels.count, r.score=Gramrels.score
            ON CREATE SET r.corpus_ententen13_tt2_1=r.corpus+$corpus
            ON CREATE SET r.GramRelPrimjerLink_ententen13_tt2_1='https://api.sketchengine.co.uk/bonito/run.cgi/view?q=w'+Gramrels.seek+';username=bperak&api_key=VZSUO4L8NV5Y9WVNWNRFU2JRRXKGTK3E&corpname='+{corpus}
    )
    
    	//to lemma
    WITH Gramrels AS gramrelSelected
    UNWIND gramrelSelected.Words AS Words
    
    	//stvori to lemme iz WSketcha i poveži s lemmafrom
    FOREACH (
    word IN CASE WHEN Words.lempos <> "" THEN [1] ELSE [] END  | 
        MERGE (to:Lemma{lempos:Words.lempos, language:{language}}) ON CREATE SET to.name=Words.word
    )
    WITH gramrelSelected, Words
    MATCH (to:Lemma{lempos:Words.lempos, language:{language}})
    MATCH(lemmafrom:Lemma{lempos:{lemmaOdabrana}+{pos}, language:{language}})
    WITH lemmafrom, to, gramrelSelected, Words
    
    CALL apoc.merge.relationship(lemmafrom, gramrelSelected.name,{relID:Words.id},{type:gramrelSelected.name, count_ententen13_tt2_1:Words.count,score_ententen13_tt2_1:Words.score, cm_ententen13_tt2_1:Words.cm, seek_ententen13_tt2_1:Words.seek, relID:Words.id, exampleLink_ententen13_tt2_1:'https://api.sketchengine.co.uk/bonito/run.cgi/view?q=w'+Words.seek+';username=bperak&api_key=VZSUO4L8NV5Y9WVNWNRFU2JRRXKGTK3E&corpname='+{corpus},language:{language}}, to) YIELD rel
    RETURN lemmafrom.name as From, type(rel), to.name as To, rel.count_ententen13_tt2_1 as Freq, rel.score_ententen13_tt2_1 as Sq 
    ORDER BY type(rel), Freq, Sq DESC
    """
        graph.run(query, userName=userName, apiKey=apiKey, metoda=metoda, lemmaOdabrana=lemmaOdabrana, pos=pos, corpus=corpus, maxitems=maxitems, language=language) #parametri
    
        for lemmaOdabrana in lemmeOdabrane:
            query2="""
            MATCH (n:Lemma{lempos:{lemmaOdabrana}+{pos},language:{language}})-[r:`"%w" and/or ...`]-(m:Lemma) 
            WHERE m.lempos ENDS WITH {pos} AND NOT EXISTS (m.freq_ententen13_tt2_1)
            RETURN m.name AS lemmaNova
            """
            data2 = graph.run(query2, userName=userName, apiKey=apiKey, lemmaOdabrana=lemmaOdabrana, pos=pos, language=language)
            for da in data2:
                print(da) #ovako izgleda json rezultat nove leme
                dar=(json.dumps(da[0], ensure_ascii=False)) #json dump izvornih znakova
                #print(dar)
                darStrip=dar.strip('"') #moraju se skinuti navodnici da ide u loop
                #provjera postoji li već sketch za lemm
                #u u bazi preko 
                #queryCheck="""
                #MATCH (n:Lemma{name:lempos:{darStrip}+{pos}}) where exists n.pos
                #"""
                #dataCheck=graph.run(queryCheck, darStrip=darStrip, pos=pos)
                
                sleep(3)
                graph.run(query, metoda=metoda, userName=userName, apiKey=apiKey, lemmaOdabrana=darStrip, pos=pos, corpus=corpus, maxitems=maxitems, language=language) #parametri
                sleep(3)

wordSketchEnTenTen13(lemmeOdabrane)
print("All the lemmas have been sucessfully stored in the Neo4j graph!")
