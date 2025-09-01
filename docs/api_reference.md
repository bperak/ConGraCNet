# ConGraCNet API Reference

> **Complete API documentation for all functions, classes, and modules**

## Table of Contents

1. [Core Functions Module](#core-functions-module)
2. [Sentiment Functions Module](#sentiment-functions-module)
3. [WordNet Functions Module](#wordnet-functions-module)
4. [Corpus Management Module](#corpus-management-module)
5. [Network Analysis Module](#network-analysis-module)
6. [Visualization Module](#visualization-module)
7. [Database Module](#database-module)
8. [Utility Functions](#utility-functions)
9. [Data Models](#data-models)
10. [Error Handling](#error-handling)

## Core Functions Module

### `cgcn_functions_3_6.py`

This module contains the core functionality for network construction, database operations, and analysis.

#### Database Operations

##### `source_lemma_freq(lemma, pos, corpusID, corpus, language, gramRel)`

Fetches the frequency of a source lemma from the database.

**Signature:**
```python
def source_lemma_freq(
    lemma: str, 
    pos: str, 
    corpusID: str, 
    corpus: str, 
    language: str, 
    gramRel: str
) -> pd.DataFrame
```

**Parameters:**
- `lemma` (str): Base form of the word
- `pos` (str): Part of speech
- `corpusID` (str): Corpus identifier
- `corpus` (str): Corpus name
- `language` (str): Language code (hr/en)
- `gramRel` (str): Grammatical relation type

**Returns:**
- `pd.DataFrame`: DataFrame with frequency and relative frequency columns

**Raises:**
- `ConnectionError`: If database connection fails
- `ValueError`: If parameters are invalid

**Example:**
```python
freq_data = source_lemma_freq("kuca", "N", "hrWaC22", "hrWaC22", "hr", "obj")
print(f"Frequency: {freq_data['freq'].iloc[0]}")
print(f"Relative Frequency: {freq_data['relFreq'].iloc[0]}")
```

##### `lemmaGramRels(lemma, pos, corpusID, language)`

Retrieves available grammatical relations for a given lemma.

**Signature:**
```python
def lemmaGramRels(
    lemma: str, 
    pos: str, 
    corpusID: str, 
    language: str
) -> pd.DataFrame
```

**Parameters:**
- `lemma` (str): Base form of the word
- `pos` (str): Part of speech
- `corpusID` (str): Corpus identifier
- `language` (str): Language code

**Returns:**
- `pd.DataFrame`: DataFrame with grammatical relations and counts

**Example:**
```python
gram_rels = lemmaGramRels("kuca", "N", "hrWaC22", "hr")
for _, row in gram_rels.iterrows():
    print(f"Relation: {row['gramRels']}, Count: {row['count']}")
```

##### `lemmaByGramRel(language, lemma, pos, gramRel, corpusID, measure)`

Builds a network based on grammatical relations.

**Signature:**
```python
def lemmaByGramRel(
    language: str, 
    lemma: str, 
    pos: str, 
    gramRel: str, 
    corpusID: str, 
    measure: str
) -> pd.DataFrame
```

**Parameters:**
- `language` (str): Language code
- `lemma` (str): Source lemma
- `pos` (str): Part of speech
- `gramRel` (str): Grammatical relation
- `corpusID` (str): Corpus identifier
- `measure` (str): Analysis measure ('score' or 'freq')

**Returns:**
- `pd.DataFrame`: DataFrame with network nodes and edges

**Example:**
```python
network_data = lemmaByGramRel("hr", "kuca", "N", "obj", "hrWaC22", "score")
print(f"Network has {len(network_data)} connections")
```

#### Network Construction

##### `construct_network(lemma, pos, corpusID, gramRel1, gramRel2, limit, measure)`

Constructs a complete network with specified parameters.

**Signature:**
```python
def construct_network(
    lemma: str,
    pos: str,
    corpusID: str,
    gramRel1: str,
    gramRel2: str,
    limit: int,
    measure: str
) -> nx.Graph
```

**Parameters:**
- `lemma` (str): Source lemma
- `pos` (str): Part of speech
- `corpusID` (str): Corpus identifier
- `gramRel1` (str): Primary grammatical relation
- `gramRel2` (str): Secondary grammatical relation
- `limit` (int): Maximum number of co-occurrences
- `measure` (str): Analysis measure

**Returns:**
- `nx.Graph`: NetworkX graph object

**Example:**
```python
network = construct_network("kuca", "N", "hrWaC22", "obj", "mod", 15, "score")
print(f"Network has {len(network.nodes)} nodes and {len(network.edges)} edges")
```

##### `expand_network(network, level, max_nodes)`

Expands a network to additional levels.

**Signature:**
```python
def expand_network(
    network: nx.Graph,
    level: int,
    max_nodes: int
) -> nx.Graph
```

**Parameters:**
- `network` (nx.Graph): Input network
- `level` (int): Expansion level
- `max_nodes` (int): Maximum number of nodes

**Returns:**
- `nx.Graph`: Expanded network

**Example:**
```python
expanded_network = expand_network(network, 2, 1000)
print(f"Expanded network has {len(expanded_network.nodes)} nodes")
```

#### Analysis Functions

##### `calculate_centrality_measures(network, measures)`

Calculates multiple centrality measures for network nodes.

**Signature:**
```python
def calculate_centrality_measures(
    network: nx.Graph,
    measures: List[str]
) -> Dict[str, Dict[str, float]]
```

**Parameters:**
- `network` (nx.Graph): NetworkX graph object
- `measures` (List[str]): List of centrality measures to calculate

**Returns:**
- `Dict[str, Dict[str, float]]`: Dictionary of centrality values by measure

**Supported Measures:**
- `'degree'`: Degree centrality
- `'weighted_degree'`: Weighted degree centrality
- `'betweenness'`: Betweenness centrality
- `'closeness'`: Closeness centrality
- `'eigenvector'`: Eigenvector centrality
- `'sli'`: Semi-Local Importance

**Example:**
```python
centrality_measures = calculate_centrality_measures(
    network, 
    ['degree', 'betweenness', 'eigenvector']
)

for node in network.nodes():
    print(f"Node: {node}")
    print(f"  Degree: {centrality_measures['degree'][node]:.4f}")
    print(f"  Betweenness: {centrality_measures['betweenness'][node]:.4f}")
    print(f"  Eigenvector: {centrality_measures['eigenvector'][node]:.4f}")
```

##### `detect_communities(network, algorithm, resolution)`

Detects communities in the network using specified algorithm.

**Signature:**
```python
def detect_communities(
    network: nx.Graph,
    algorithm: str = 'louvain',
    resolution: float = 1.0
) -> List[Set[str]]
```

**Parameters:**
- `network` (nx.Graph): NetworkX graph object
- `algorithm` (str): Community detection algorithm
- `resolution` (float): Resolution parameter for CPM algorithm

**Returns:**
- `List[Set[str]]`: List of community sets

**Supported Algorithms:**
- `'louvain'`: Louvain community detection
- `'leiden'`: Leiden community detection
- `'greedy'`: Greedy modularity optimization
- `'cpm'`: Constant Potts Model

**Example:**
```python
communities = detect_communities(network, algorithm='louvain')
print(f"Detected {len(communities)} communities")

for i, community in enumerate(communities):
    print(f"Community {i+1}: {len(community)} nodes")
    print(f"  Nodes: {list(community)[:5]}...")
```

## Sentiment Functions Module

### `sentiment_functions_3_6.py`

This module handles sentiment analysis using multiple dictionaries and network-based propagation.

#### Dictionary Integration

##### `get_SenticNet_concept_df(lempos, language)`

Retrieves SenticNet 6 sentiment values for a concept.

**Signature:**
```python
def get_SenticNet_concept_df(
    lempos: str,
    language: str
) -> pd.DataFrame
```

**Parameters:**
- `lempos` (str): Lemma-pos combination
- `language` (str): Language code

**Returns:**
- `pd.DataFrame`: DataFrame with SenticNet sentiment values

**Columns:**
- `label`: Concept label
- `polarity_value`: Sentiment polarity score
- `polarity_label`: Sentiment polarity label

**Example:**
```python
senticnet_data = get_SenticNet_concept_df("kuca-N", "hr")
print(f"Sentiment: {senticnet_data['polarity_label'].iloc[0]}")
print(f"Score: {senticnet_data['polarity_value'].iloc[0]}")
```

##### `get_sentiWords_values_df(lempos_list, sentiWords_dict)`

Retrieves SentiWords 1.1 sentiment values.

**Signature:**
```python
def get_sentiWords_values_df(
    lempos_list: List[str],
    sentiWords_dict: Dict
) -> pd.DataFrame
```

**Parameters:**
- `lempos_list` (List[str]): List of lemma-pos combinations
- `sentiWords_dict` (Dict): SentiWords dictionary

**Returns:**
- `pd.DataFrame`: DataFrame with SentiWords sentiment values

**Example:**
```python
sentiwords_data = get_sentiWords_values_df(["house-N", "car-N"], sentiWords)
for _, row in sentiwords_data.iterrows():
    print(f"{row['word']}: Pos={row['pos_score']:.4f}, Neg={row['neg_score']:.4f}")
```

##### `get_word_sentiWordsNet_values_df(sentiWordsNet_dict, lempos_list)`

Retrieves SentiWordNet sentiment values.

**Signature:**
```python
def get_word_sentiWordsNet_values_df(
    sentiWordsNet_dict: Dict,
    lempos_list: List[str]
) -> pd.DataFrame
```

**Parameters:**
- `sentiWordsNet_dict` (Dict): SentiWordNet dictionary
- `lempos_list` (List[str]): List of lemma-pos combinations

**Returns:**
- `pd.DataFrame`: DataFrame with SentiWordNet sentiment values

**Columns:**
- `SynsetTerms`: Synset terms
- `PosScore`: Positive sentiment score
- `NegScore`: Negative sentiment score
- `Gloss`: Word definition

#### Sentiment Calculation

##### `calculate_sentic_values(lemma, pos, language, measure, corpus, corpusID, gramRel, limit_f, limit_fof, harvest_select)`

Calculates network-based SenticNet values.

**Signature:**
```python
def calculate_sentic_values(
    lemma: List[str],
    pos: str,
    language: str,
    measure: str,
    corpus: str,
    corpusID: str,
    gramRel: str,
    limit_f: int,
    limit_fof: int,
    harvest_select: str
) -> pd.DataFrame
```

**Parameters:**
- `lemma` (List[str]): List of lemmas
- `pos` (str): Part of speech
- `language` (str): Language code
- `measure` (str): Analysis measure
- `corpus` (str): Corpus name
- `corpusID` (str): Corpus identifier
- `gramRel` (str): Grammatical relation
- `limit_f` (int): Limit for friends
- `limit_fof` (int): Limit for friends-of-friends
- `harvest_select` (str): Harvest selection ('y' or 'n')

**Returns:**
- `pd.DataFrame`: DataFrame with calculated sentiment values

**Example:**
```python
sentic_values = calculate_sentic_values(
    ["kuca"], "N", "hr", "score", "hrWaC22", "hrWaC22", 
    "obj", 15, 5, "y"
)
print(f"Calculated sentiment values for {len(sentic_values)} concepts")
```

##### `calculate_sentiWords_values(sentiWords_dict, lemma, pos, language, measure, corpus, corpusID, gramRel, limit_f, limit_fof, harvest_select)`

Calculates network-based SentiWords values.

**Signature:**
```python
def calculate_sentiWords_values(
    sentiWords_dict: Dict,
    lemma: List[str],
    pos: str,
    language: str,
    measure: str,
    corpus: str,
    corpusID: str,
    gramRel: str,
    limit_f: int,
    limit_fof: int,
    harvest_select: str
) -> pd.DataFrame
```

**Parameters:**
- `sentiWords_dict` (Dict): SentiWords dictionary
- `lemma` (List[str]): List of lemmas
- `pos` (str): Part of speech
- `language` (str): Language code
- `measure` (str): Analysis measure
- `corpus` (str): Corpus name
- `corpusID` (str): Corpus identifier
- `gramRel` (str): Grammatical relation
- `limit_f` (int): Limit for friends
- `limit_fof` (int): Limit for friends-of-friends
- `harvest_select` (str): Harvest selection

**Returns:**
- `pd.DataFrame`: DataFrame with calculated SentiWords values

#### Sentiment Propagation

##### `propagate_sentiment_network(network, initial_sentiment, method, decay_factor)`

Propagates sentiment through network structure.

**Signature:**
```python
def propagate_sentiment_network(
    network: nx.Graph,
    initial_sentiment: Dict[str, float],
    method: str = 'linear',
    decay_factor: float = 0.5
) -> Dict[str, float]
```

**Parameters:**
- `network` (nx.Graph): NetworkX graph object
- `initial_sentiment` (Dict[str, float]): Initial sentiment values
- `method` (str): Propagation method ('linear' or 'exponential')
- `decay_factor` (float): Sentiment decay factor

**Returns:**
- `Dict[str, float]`: Updated sentiment values for all nodes

**Example:**
```python
# Initial sentiment for source lemma
initial_sentiment = {"kuca-N": 0.8}

# Propagate sentiment through network
propagated_sentiment = propagate_sentiment_network(
    network, 
    initial_sentiment, 
    method='linear', 
    decay_factor=0.6
)

# Display results
for node, sentiment in propagated_sentiment.items():
    print(f"{node}: {sentiment:.4f}")
```

## WordNet Functions Module

### `wordnet_functions.py`

This module provides WordNet integration for lexical-semantic relationships.

#### Synset Management

##### `get_hypernyms(lempos_list, language)`

Extracts hypernym relationships for given lemmas.

**Signature:**
```python
def get_hypernyms(
    lempos_list: List[str],
    language: str
) -> pd.DataFrame
```

**Parameters:**
- `lempos_list` (List[str]): List of lemma-pos combinations
- `language` (str): Language code

**Returns:**
- `pd.DataFrame`: DataFrame with hypernym relationships

**Columns:**
- `source`: Source lemma
- `target`: Hypernym lemma
- `s_definition`: Source definition
- `t_definition`: Target definition
- `valence_s`: Source valence
- `valence_t`: Target valence

**Example:**
```python
hypernyms = get_hypernyms(["kuca-N", "auto-N"], "hr")
for _, row in hypernyms.iterrows():
    print(f"{row['source']} -> {row['target']}")
    print(f"  Source def: {row['s_definition']}")
    print(f"  Target def: {row['t_definition']}")
```

##### `get_antonyms_from_lempos(lempos_list, language)`

Retrieves antonyms for given lemma-pos combinations.

**Signature:**
```python
def get_antonyms_from_lempos(
    lempos_list: List[str],
    language: str
) -> pd.DataFrame
```

**Parameters:**
- `lempos_list` (List[str]): List of lemma-pos combinations
- `language` (str): Language code

**Returns:**
- `pd.DataFrame`: DataFrame with antonym relationships

**Example:**
```python
antonyms = get_antonyms_from_lempos(["velika-ADJ", "mala-ADJ"], "hr")
for _, row in antonyms.iterrows():
    print(f"Antonym pair: {row['source']} - {row['target']}")
```

#### Valence Functions

##### `get_valency_for_lempos(lempos_list)`

Retrieves valence values from VADER lexicon.

**Signature:**
```python
def get_valency_for_lempos(
    lempos_list: List[str]
) -> Dict[str, Dict[str, float]]
```

**Parameters:**
- `lempos_list` (List[str]): List of lemma-pos combinations

**Returns:**
- `Dict[str, Dict[str, float]]`: Dictionary of valence values by lemma

**Valence Components:**
- `pos`: Positive valence
- `neg`: Negative valence
- `neu`: Neutral valence
- `compound`: Compound valence score

**Example:**
```python
valence_data = get_valency_for_lempos(["house", "car", "computer"])
for lemma, valence in valence_data.items():
    print(f"{lemma}:")
    print(f"  Positive: {valence['pos']:.4f}")
    print(f"  Negative: {valence['neg']:.4f}")
    print(f"  Neutral: {valence['neu']:.4f}")
    print(f"  Compound: {valence['compound']:.4f}")
```

### `spacy_wordnet_functions.py`

This module provides spaCy-WordNet integration for domain classification.

#### Domain Functions

##### `get_domains_for_word(word)`

Gets semantic domains for a given word.

**Signature:**
```python
def get_domains_for_word(word: str) -> List[str]
```

**Parameters:**
- `word` (str): Target word

**Returns:**
- `List[str]`: List of semantic domain categories

**Example:**
```python
domains = get_domains_for_word("kuca")
print(f"Domains for 'kuca': {', '.join(domains)}")
```

## Corpus Management Module

### `sketch2Neo2.py`

This module manages corpus selection and Sketch Engine API integration.

#### Corpus Functions

##### `corpusSelect(corpus_selection)`

Selects corpus based on user selection.

**Signature:**
```python
def corpusSelect(corpus_selection: str) -> Tuple[str, str, str, str, str, str]
```

**Parameters:**
- `corpus_selection` (str): Corpus selection identifier

**Returns:**
- `Tuple[str, str, str, str, str, str]`: Corpus information tuple
  - Corpus name
  - Language
  - Corpus ID
  - Initial grammatical relation
  - Initial lexeme
  - Initial part of speech

**Example:**
```python
corpus_info = corpusSelect("hrWaC22")
corpus_name, language, corpus_id, gram_rel, lexeme, pos = corpus_info
print(f"Selected corpus: {corpus_name} ({language})")
print(f"Initial lexeme: {lexeme}-{pos}")
```

##### `corpusSketchHarvest(corpus_selection, lemma_list, username, api_key, endpoint, pos1, pos2, corpus, limit, language, gram_rel)`

Harvests corpus data from Sketch Engine API.

**Signature:**
```python
def corpusSketchHarvest(
    corpus_selection: str,
    lemma_list: List[str],
    username: str,
    api_key: str,
    endpoint: str,
    pos1: str,
    pos2: str,
    corpus: str,
    limit: int,
    language: str,
    gram_rel: str
) -> str
```

**Parameters:**
- `corpus_selection` (str): Corpus selection
- `lemma_list` (List[str]): List of lemmas to harvest
- `username` (str): Sketch Engine username
- `api_key` (str): Sketch Engine API key
- `endpoint` (str): API endpoint
- `pos1` (str): First part of speech
- `pos2` (str): Second part of speech
- `corpus` (str): Corpus name
- `limit` (int): Maximum results
- `language` (str): Language code
- `gram_rel` (str): Grammatical relation

**Returns:**
- `str`: Harvest result status

**Example:**
```python
result = corpusSketchHarvest(
    "hrWaC22", ["kuca", "auto"], "username", "api_key",
    "wsketch?", "N", "N", "hrWaC22", 15, "hr", "obj"
)
print(f"Harvest result: {result}")
```

#### Database Functions

##### `createIndexes()`

Creates database indexes for performance optimization.

**Signature:**
```python
def createIndexes() -> None
```

**Example:**
```python
createIndexes()
print("Database indexes created successfully")
```

## Network Analysis Module

### Network Construction

#### `build_network_from_data(network_data, directed)`

Builds NetworkX graph from network data.

**Signature:**
```python
def build_network_from_data(
    network_data: pd.DataFrame,
    directed: bool = False
) -> nx.Graph
```

**Parameters:**
- `network_data` (pd.DataFrame): Network data DataFrame
- `directed` (bool): Whether to create directed graph

**Returns:**
- `nx.Graph`: NetworkX graph object

**Example:**
```python
network = build_network_from_data(network_data, directed=False)
print(f"Created {'directed' if network.is_directed() else 'undirected'} graph")
```

#### `prune_network(network, min_freq, min_score, percentile)`

Prunes network based on frequency and score thresholds.

**Signature:**
```python
def prune_network(
    network: nx.Graph,
    min_freq: Optional[int] = None,
    min_score: Optional[float] = None,
    percentile: Optional[float] = None
) -> nx.Graph
```

**Parameters:**
- `network` (nx.Graph): Input network
- `min_freq` (Optional[int]): Minimum frequency threshold
- `min_score` (Optional[float]): Minimum score threshold
- `percentile` (Optional[float]): Percentile threshold for pruning

**Returns:**
- `nx.Graph`: Pruned network

**Example:**
```python
pruned_network = prune_network(
    network, 
    min_freq=5, 
    min_score=0.1, 
    percentile=90
)
print(f"Pruned network: {len(pruned_network.nodes)} nodes, {len(pruned_network.edges)} edges")
```

### Centrality Analysis

#### `calculate_sli_centrality(network, k)`

Calculates Semi-Local Importance centrality.

**Signature:**
```python
def calculate_sli_centrality(
    network: nx.Graph,
    k: int = 2
) -> Dict[str, float]
```

**Parameters:**
- `network` (nx.Graph): NetworkX graph object
- `k` (int): Neighborhood depth for SLI calculation

**Returns:**
- `Dict[str, float]`: SLI centrality values by node

**Example:**
```python
sli_centrality = calculate_sli_centrality(network, k=3)
for node, centrality in sorted(sli_centrality.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"{node}: {centrality:.4f}")
```

### Community Detection

#### `detect_communities_louvain(network)`

Detects communities using Louvain algorithm.

**Signature:**
```python
def detect_communities_louvain(network: nx.Graph) -> List[Set[str]]
```

**Parameters:**
- `network` (nx.Graph): NetworkX graph object

**Returns:**
- `List[Set[str]]`: List of community sets

**Example:**
```python
communities = detect_communities_louvain(network)
print(f"Louvain detected {len(communities)} communities")

modularity = nx.community.modularity(network, communities)
print(f"Modularity: {modularity:.4f}")
```

#### `detect_communities_leiden(network)`

Detects communities using Leiden algorithm.

**Signature:**
```python
def detect_communities_leiden(network: nx.Graph) -> List[Set[str]]
```

**Parameters:**
- `network` (nx.Graph): NetworkX graph object

**Returns:**
- `List[Set[str]]`: List of community sets

**Example:**
```python
communities = detect_communities_leiden(network)
print(f"Leiden detected {len(communities)} communities")

# Calculate quality metrics
modularity = nx.community.modularity(network, communities)
coverage = sum(len(comm) for comm in communities) / len(network)
print(f"Modularity: {modularity:.4f}, Coverage: {coverage:.4f}")
```

## Visualization Module

### Network Visualization

#### `create_network_plot(network, layout, node_size, edge_width, labels)`

Creates interactive network visualization using Plotly.

**Signature:**
```python
def create_network_plot(
    network: nx.Graph,
    layout: str = 'spring',
    node_size: Union[str, float] = 10,
    edge_width: Union[str, float] = 1,
    labels: bool = True
) -> go.Figure
```

**Parameters:**
- `network` (nx.Graph): NetworkX graph object
- `layout` (str): Layout algorithm ('spring', 'circular', 'random', 'kamada_kawai')
- `node_size` (Union[str, float]): Node size (attribute name or fixed value)
- `edge_width` (Union[str, float]): Edge width (attribute name or fixed value)
- `labels` (bool): Whether to show node labels

**Returns:**
- `go.Figure`: Plotly figure object

**Example:**
```python
# Create visualization with degree-based node sizes
fig = create_network_plot(
    network,
    layout='spring',
    node_size='degree',
    edge_width='weight',
    labels=True
)

# Display the plot
fig.show()
```

#### `create_3d_network_plot(network, layout, node_size, edge_width)`

Creates 3D network visualization.

**Signature:**
```python
def create_3d_network_plot(
    network: nx.Graph,
    layout: str = 'spring',
    node_size: Union[str, float] = 10,
    edge_width: Union[str, float] = 1
) -> go.Figure
```

**Parameters:**
- `network` (nx.Graph): NetworkX graph object
- `layout` (str): 3D layout algorithm
- `node_size` (Union[str, float]): Node size
- `edge_width` (Union[str, float]): Edge width

**Returns:**
- `go.Figure`: 3D Plotly figure object

**Example:**
```python
fig_3d = create_3d_network_plot(
    network,
    layout='spring',
    node_size='degree',
    edge_width='weight'
)

# Display 3D plot
fig_3d.show()
```

### Statistical Visualizations

#### `create_centrality_comparison_plot(centrality_data)`

Creates comparison plot of different centrality measures.

**Signature:**
```python
def create_centrality_comparison_plot(
    centrality_data: Dict[str, Dict[str, float]]
) -> go.Figure
```

**Parameters:**
- `centrality_data` (Dict[str, Dict[str, float]]): Centrality data by measure

**Returns:**
- `go.Figure`: Plotly figure with centrality comparison

**Example:**
```python
centrality_data = {
    'degree': degree_centrality,
    'betweenness': betweenness_centrality,
    'eigenvector': eigenvector_centrality
}

fig = create_centrality_comparison_plot(centrality_data)
fig.show()
```

#### `create_community_analysis_plot(network, communities)`

Creates community analysis visualization.

**Signature:**
```python
def create_community_analysis_plot(
    network: nx.Graph,
    communities: List[Set[str]]
) -> go.Figure
```

**Parameters:**
- `network` (nx.Graph): NetworkX graph object
- `communities` (List[Set[str]]): List of community sets

**Returns:**
- `go.Figure`: Plotly figure with community analysis

**Example:**
```python
fig = create_community_analysis_plot(network, communities)
fig.show()
```

## Database Module

### Connection Management

#### `init_neo4j()`

Initializes Neo4j database connection.

**Signature:**
```python
def init_neo4j() -> Optional[Graph]
```

**Returns:**
- `Optional[Graph]`: Neo4j graph object or None if connection fails

**Example:**
```python
graph = init_neo4j()
if graph is None:
    print("Failed to connect to Neo4j database")
else:
    print("Successfully connected to Neo4j database")
```

### Query Functions

#### `run_cypher_query(query, parameters)`

Executes Cypher query with parameters.

**Signature:**
```python
def run_cypher_query(
    query: str,
    parameters: Dict[str, Any]
) -> pd.DataFrame
```

**Parameters:**
- `query` (str): Cypher query string
- `parameters` (Dict[str, Any]): Query parameters

**Returns:**
- `pd.DataFrame`: Query results as DataFrame

**Example:**
```python
query = """
MATCH (n:Lemma {lempos: $lempos})
WHERE n[$freq] > $min_freq
RETURN n.lempos as lemma, n[$freq] as frequency
ORDER BY frequency DESC
LIMIT 10
"""

params = {
    'lempos': 'kuca-N',
    'freq': 'freq_hrWaC22',
    'min_freq': 5
}

results = run_cypher_query(query, params)
print(f"Found {len(results)} high-frequency lemmas")
```

#### `batch_query_execution(queries, parameters_list)`

Executes multiple queries in batch.

**Signature:**
```python
def batch_query_execution(
    queries: List[str],
    parameters_list: List[Dict[str, Any]]
) -> List[pd.DataFrame]
```

**Parameters:**
- `queries` (List[str]): List of Cypher queries
- `parameters_list` (List[Dict[str, Any]]): List of parameter dictionaries

**Returns:**
- `List[pd.DataFrame]`: List of query results

**Example:**
```python
queries = [
    "MATCH (n:Lemma {lempos: $lempos1}) RETURN n.lempos as lemma",
    "MATCH (n:Lemma {lempos: $lempos2}) RETURN n.lempos as lemma"
]

params_list = [
    {'lempos1': 'kuca-N'},
    {'lempos2': 'auto-N'}
]

results = batch_query_execution(queries, params_list)
for i, result in enumerate(results):
    print(f"Query {i+1} returned {len(result)} results")
```

## Utility Functions

### Data Processing

#### `col_order(df, cols_list)`

Reorganizes DataFrame columns according to specified order.

**Signature:**
```python
def col_order(
    df: pd.DataFrame,
    cols_list: List[str]
) -> pd.DataFrame
```

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `cols_list` (List[str]): Desired column order

**Returns:**
- `pd.DataFrame`: DataFrame with reordered columns

**Example:**
```python
# Reorder columns with important ones first
reordered_df = col_order(df, ['lemma', 'frequency', 'score', 'pos'])
print(f"Column order: {list(reordered_df.columns)}")
```

#### `validate_network_data(network_data)`

Validates network data structure and content.

**Signature:**
```python
def validate_network_data(
    network_data: pd.DataFrame
) -> Tuple[bool, List[str]]
```

**Parameters:**
- `network_data` (pd.DataFrame): Network data to validate

**Returns:**
- `Tuple[bool, List[str]]`: Validation result and error messages

**Example:**
```python
is_valid, errors = validate_network_data(network_data)
if is_valid:
    print("Network data is valid")
else:
    print("Validation errors found:")
    for error in errors:
        print(f"  - {error}")
```

### Performance Utilities

#### `measure_execution_time(func, *args, **kwargs)`

Measures execution time of a function.

**Signature:**
```python
def measure_execution_time(
    func: Callable,
    *args,
    **kwargs
) -> Tuple[Any, float]
```

**Parameters:**
- `func` (Callable): Function to measure
- `*args`: Positional arguments
- `**kwargs`: Keyword arguments

**Returns:**
- `Tuple[Any, float]`: Function result and execution time

**Example:**
```python
result, execution_time = measure_execution_time(
    detect_communities, 
    network, 
    algorithm='louvain'
)
print(f"Community detection completed in {execution_time:.2f} seconds")
```

#### `memory_usage_monitor()`

Monitors memory usage of the application.

**Signature:**
```python
def memory_usage_monitor() -> Dict[str, float]
```

**Returns:**
- `Dict[str, float]`: Memory usage statistics

**Example:**
```python
memory_stats = memory_usage_monitor()
print(f"Memory usage: {memory_stats['used_mb']:.1f} MB")
print(f"Memory available: {memory_stats['available_mb']:.1f} MB")
print(f"Memory percentage: {memory_stats['percent']:.1f}%")
```

## Data Models

### Network Data Structure

#### Network DataFrame
```python
# Expected structure for network data
network_data = pd.DataFrame({
    'source': ['kuca-N', 'kuca-N', 'auto-N'],      # Source lemma-pos
    'target': ['velika-ADJ', 'mala-ADJ', 'brz-ADJ'], # Target lemma-pos
    'gramRel': ['obj', 'obj', 'mod'],               # Grammatical relation
    'count': [150, 120, 80],                        # Co-occurrence count
    'score': [0.85, 0.72, 0.65],                   # Association score
    'corpus': ['hrWaC22', 'hrWaC22', 'hrWaC22']    # Corpus identifier
})
```

#### Centrality Data Structure
```python
# Centrality measures data structure
centrality_data = {
    'degree': {
        'kuca-N': 0.85,
        'velika-ADJ': 0.72,
        'mala-ADJ': 0.65
    },
    'betweenness': {
        'kuca-N': 0.45,
        'velika-ADJ': 0.23,
        'mala-ADJ': 0.18
    },
    'eigenvector': {
        'kuca-N': 0.92,
        'velika-ADJ': 0.68,
        'mala-ADJ': 0.54
    }
}
```

#### Community Data Structure
```python
# Community detection results
communities = [
    {'kuca-N', 'velika-ADJ', 'mala-ADJ', 'lijepa-ADJ'},  # Community 1
    {'auto-N', 'brz-ADJ', 'nov-ADJ', 'star-ADJ'},        # Community 2
    {'grad-N', 'velik-ADJ', 'lijep-ADJ', 'moderan-ADJ'}  # Community 3
]
```

### Sentiment Data Structure

#### Sentiment Values
```python
# Sentiment analysis results
sentiment_data = {
    'kuca-N': {
        'original': 0.75,      # Original dictionary value
        'assigned': 0.82,      # Network-assigned value
        'difference': 0.07,    # Difference between values
        'confidence': 0.89     # Confidence in assigned value
    },
    'velika-ADJ': {
        'original': 0.45,
        'assigned': 0.51,
        'difference': 0.06,
        'confidence': 0.76
    }
}
```

## Error Handling

### Custom Exception Classes

#### `ConGraCNetError`
Base exception class for ConGraCNet application.

```python
class ConGraCNetError(Exception):
    """Base exception for ConGraCNet application."""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)
```

#### `DatabaseConnectionError`
Raised when database connection fails.

```python
class DatabaseConnectionError(ConGraCNetError):
    """Raised when database connection fails."""
    
    def __init__(self, message: str, database_url: Optional[str] = None):
        self.database_url = database_url
        super().__init__(f"Database connection failed: {message}")
```

#### `APIError`
Raised when external API calls fail.

```python
class APIError(ConGraCNetError):
    """Raised when external API calls fail."""
    
    def __init__(self, message: str, api_endpoint: Optional[str] = None, status_code: Optional[int] = None):
        self.api_endpoint = api_endpoint
        self.status_code = status_code
        super().__init__(f"API error: {message}")
```

#### `NetworkConstructionError`
Raised when network construction fails.

```python
class NetworkConstructionError(ConGraCNetError):
    """Raised when network construction fails."""
    
    def __init__(self, message: str, lemma: Optional[str] = None, corpus: Optional[str] = None):
        self.lemma = lemma
        self.corpus = corpus
        super().__init__(f"Network construction failed: {message}")
```

### Error Handling Patterns

#### Function-Level Error Handling
```python
def safe_network_construction(lemma: str, pos: str, corpus: str) -> nx.Graph:
    """
    Safely constructs network with comprehensive error handling.
    
    Args:
        lemma: Source lemma
        pos: Part of speech
        corpus: Corpus identifier
        
    Returns:
        NetworkX graph object
        
    Raises:
        NetworkConstructionError: If construction fails
        DatabaseConnectionError: If database is unavailable
    """
    try:
        # Validate inputs
        if not lemma or not pos or not corpus:
            raise ValueError("All parameters must be provided")
        
        # Attempt network construction
        network = construct_network(lemma, pos, corpus)
        
        if network is None or len(network.nodes) == 0:
            raise NetworkConstructionError("Network construction returned empty result", lemma, corpus)
        
        return network
        
    except ValueError as e:
        raise NetworkConstructionError(f"Invalid parameters: {str(e)}", lemma, corpus)
    except ConnectionError as e:
        raise DatabaseConnectionError(f"Database connection failed: {str(e)}")
    except Exception as e:
        raise NetworkConstructionError(f"Unexpected error: {str(e)}", lemma, corpus)
```

#### Application-Level Error Handling
```python
def handle_application_error(error: Exception, context: Dict[str, Any]) -> None:
    """
    Handles application-level errors with logging and user feedback.
    
    Args:
        error: Exception that occurred
        context: Context information about the error
    """
    error_id = str(uuid.uuid4())
    
    # Log error details
    logging.error(f"Error ID: {error_id}")
    logging.error(f"Error type: {type(error).__name__}")
    logging.error(f"Error message: {str(error)}")
    logging.error(f"Context: {context}")
    
    # Provide user-friendly error message
    if isinstance(error, DatabaseConnectionError):
        st.error(f"Database connection failed. Please check your connection settings. Error ID: {error_id}")
    elif isinstance(error, APIError):
        st.error(f"API request failed. Please try again later. Error ID: {error_id}")
    elif isinstance(error, NetworkConstructionError):
        st.error(f"Network construction failed. Please check your parameters. Error ID: {error_id}")
    else:
        st.error(f"An unexpected error occurred. Error ID: {error_id}")
    
    # Display error details in expander for debugging
    with st.expander("Error Details (for debugging)"):
        st.code(f"""
Error ID: {error_id}
Error Type: {type(error).__name__}
Error Message: {str(error)}
Context: {json.dumps(context, indent=2)}
        """)
```

---

*This API reference provides comprehensive documentation for all functions and modules in ConGraCNet. For usage examples and tutorials, see the User Manual and Developer Guide.*
