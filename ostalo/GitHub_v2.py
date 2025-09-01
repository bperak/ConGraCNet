def sli_importance(G, **kwargs):
    '''
    The algorithm takes graph G and get the importance of the nodes as a list of floating values
    kwargs: 
        igraph (default False), if True transforms G from igraph to Networkx
        normalize (default True), if False returns non-normalized values from a list of SLI importance values
    '''
    # G introduced in igraph module is transformed into networkx module. If you want to introduce this feature you have to import igraph module
    if kwargs.get('igraph')==True:
        # import igraph
        g=G.to_networkx()
    else:
        g=G
    
    #Detection of basic cycles
    cycle = nx.cycle_basis(g.to_undirected()) #Will contain: node_value  -> sum_of_degrees
    degrees = {}    
    dict1 = {}
    for connect in (nx.edges(g)):
        degrees[connect[0]] = degrees.get(connect[0], 0) + g.degree(connect[1])
        degrees[connect[1]] = degrees.get(connect[1], 0) + g.degree(connect[0])

    #For each edge we get if a basic cycle contains it
        for cicl in cycle:
            in_path = lambda e, path: (e[0], e[1]) in path or (e[1], e[0]) in path
            cycle_to_path = lambda path: list(zip(path+path[:1], path[1:] + path[:1]))
            in_a_cycle = lambda e, cycle: in_path(e, cycle_to_path(cycle))
            in_any_cycle = lambda e, g: any(in_a_cycle(e, c) for c in nx.cycle_basis(g))

        counter_connect=0
        
    #For each edge the number of basic cycles that contain it is calculated
        if in_any_cycle(connect, g)==True:
            for cicl in (cycle):
                c=set(cicl)
                set1=set(connect)
                set2=set(c)
                is_subset = set1.issubset(set2)
                if is_subset==True:
                    counter_connect+=1
    
        dict1[connect] = counter_connect
        dict1[list(itertools.permutations(connect))[1]] = counter_connect
    
    
    SLI_importance = []
    for node in nx.nodes(g):
        sum = 0
        for neigh in g.neighbors(node):           
            edge_weight= g.get_edge_data(node,neigh)['weight']
            nodeWeight_node = G.vs[node]["weighted_degree"]
            nodeWeight_neigh =  G.vs[neigh]["weighted_degree"]
            p = dict1[(node, neigh)] 
            u = (nodeWeight_node+ nodeWeight_neigh - 2* edge_weight)
            lambd = p+1
            z = nodeWeight_node/ (nodeWeight_node+ nodeWeight_neigh)*edge_weight
            I = u*lambd*z
            sum = sum + I
        SLI_importance.append(sum + nodeWeight_node)
    SLI_importance= pd.Series(SLI_importance)
    
    
    # SLI values non-normalized
    if kwargs.get('normalize') == False:
        SLI_importance_result = SLI_importance
    # SLI values normalized as default
    else:
        SLI_importance_normalized = SLI_importance/SLI_importance.sum()*100
        SLI_importance_result = SLI_importance_normalized 
    return SLI_importance_result
