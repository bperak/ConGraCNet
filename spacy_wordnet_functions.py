#%% This function extracts domains for a word 
import spacy
from spacy_wordnet.wordnet_annotator import WordnetAnnotator 

import subprocess

try:
    # Try to load the model
    nlp = spacy.load('en_core_web_sm')
    print(f"en_core_web_sm loaded.")
except OSError:
    # If model is not found, download it
    print(f"en_core_web_sm not found. Downloading...")
    command = "python -m spacy download en_core_web_sm"
    subprocess.call(command, shell=True)
    print(f"en_core_web_sm is now installed.Try again")


try:
    nlp.add_pipe(WordnetAnnotator(nlp.lang), after='tagger')
except:
    nlp.add_pipe("spacy_wordnet", after='tagger')
finally:
    pass 
#, config={'lang': nlp.lang}
#%%
def get_domains_for_word(word):
    '''Get domains for a word'''
    try:
        word= word.split('.')[0]
        token = nlp(word)[0]
        # wordnet object link spacy token with nltk wordnet interface by giving acces to
        # synsets and lemmas 
        # token._.wordnet.synsets()
        # token._.wordnet.lemmas()
        domains= token._.wordnet.wordnet_domains()
    # And automatically tags with wordnet domains
    except:
        pass
    return (domains)

def get_domains_for_synsets(sysnsetList):
    wndomains=[]
    for word in sysnsetList:
        word= word.split('.')[0]
        token = nlp(word)[0]
        # wordnet object link spacy token with nltk wordnet interface by giving acces to
        # synsets and lemmas 
        token._.wordnet.synsets()
        token._.wordnet.lemmas()
        wndomains.append(token._.wordnet.wordnet_domains())
    # And automatically tags with wordnet domains
    return (wndomains)

# get_domains_for_word('frequency')
#%% get_words_in_sent
def get_words_in_sent(sentence, domains_list):
    # Imagine we want to enrich the following sentence with synonyms
    sentence = nlp(sentence)

    # spaCy WordNet lets you find synonyms by domain of interest
    enriched_sentence = []

    # For each token in the sentence
    for token in sentence:
        # We get those synsets within the desired domains
        synsets = token._.wordnet.wordnet_synsets_for_domain(domains_list)
        if synsets:
            lemmas_for_synset = []
            for s in synsets:
                # If we found a synset in the economy domains
                # we get the variants and add them to the enriched sentence
                lemmas_for_synset.extend(s.lemma_names())
                enriched_sentence.append('({})'.format('|'.join(set(lemmas_for_synset))))
        else:
            enriched_sentence.append(token.text)

    # Let's see our enriched sentence
    return (' '.join(enriched_sentence))
