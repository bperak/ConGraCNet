<details>

<summary style="text-align:right;font-size:15px"><b>Labeling parameters</b></summary><a name="labeling-s"></a>

> This algorithm uses *is-a* construction to find more abstract categories that can be used to label the community. The *is-a* network is constructed from 15 *N_source(1,2...n) is-a N_category* collocations. The algorithm:  
* For each lemma in a *Pruned graph cluster* identify `is-a` friends (target). The network is *directed*. We expect the more abstract *category* to appear in the is-a target node.  
* Create is-a target friends network and identify the central node. 
We expect the *category* to appear as the central node of the FoF network. 

>The eigenvector centrality is proposed as the most optimal algorithm to identify the most representative categories. However, you can choose to rank nodes by different measures with `Sort is_a dataset by` selection. 


</details>

