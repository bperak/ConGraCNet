<style>
summary::-webkit-details-marker {
  display: none;
  }
summary {
    padding-left: 10px;
}
    </style>

<details >
<summary style="text-align:left;font-size:25px; padding-left:20px"><img src="http://emocnet.uniri.hr/wp-content/uploads/2020/09/statistics-300x300.png" alt="ConGraCNet info" width=30><b>&nbsp;&nbsp;ConGraCNet</b></summary>


> ### Corpus-based graph app
Construction Grammar Conceptual Networks is a corpus-based graph approach to the syntactic-semantic analysis of concepts. The application uses the syntactic relations of a syntactically tagged corpora to represent various semantic relations in terms of a network structure. 
### Corpus selection
The data is provided from the *Sketch Engine * using the API. You can choose a corpus from the available corpora in from the `Select the corpus` selection box. 
### Lemma and pos selection
You can choose a lexeme in the `Select source lemma` box along with its part of the speech `Select source pos` the set of grammatical relations is revealed.
The summary about the source lexeme is revealed in the `Source lexeme` section. It displays the `Frequency in corpus`, `Relative frequency in corpus` and count of available grammar relations in the `Grammatical relations list`.
### Network construction parameters
You can create a network by choosing a number of coocurrences set in the `Friend coocurences`. The coocurrences are acquired by a measure set in the `Measure` selection box. If there are not enough available data about the source word in the local database you can harvest the data by checking `Initial data harvest`. Similarly, you can harvest coocurrences data by checking `Friend harvesting option`. You can set additional parameters by checking `Set minimal frequency and score`.
### Pruning
Pruning parameters determine the manner of the pruning and the size  of the second degree graph. You can select `Betweenness percentile` parameters, `Degree pruned` and `Partition type for pruned graph`. If you choose a `cpm` option, an additional slider with `Resolution for pruned graph (0-low, 1-high)` is enabled, giving you the option to refine or engross the communities in the pruned graph.
### Clustering
For a selected source word the structure of the semantic domains is revealed as a set of sub-graphs derived from the source lexeme syntactic relations.  The sub-graph structures, calculated with the community detection algorithm, are interpreted as the semantic domains associated with the conceptual matrix of the source lexeme.
### Labeling 
This algorithm uses is-a construction to find more abstract categories that can be used to label the community. The is-a network is constructed from 15 N_source(1,2...n) is-a N_category collocations. The algorithm:

>* For each lemma in a Pruned graph cluster identify is-a friends (target). The network is directed. We expect the more abstract category to appear in the is-a target node.
>* Create is-a target friends network and identify the central node. We expect the category to appear as the central node of the FoF network.
The eigenvector centrality is proposed as the most optimal algorithm to identify the most representative categories. However, you can choose to rank nodes by different measures with Sort is_a dataset by selection.

</details>



