import networkx as nx

def count_cycles_containing_edges(G):
    cycles = list(nx.simple_cycles(G))
    print(cycles)
    edge_cycle_count = {edge: 0 for edge in G.edges()}
    for edge in G.edges():
        for cycle in cycles:
            # Consider both directions for undirected comparison, if needed
            cycle_edges = list(zip(cycle, cycle[1:] + cycle[:1])) + list(zip(cycle[-1:] + cycle[:-1], cycle))
            if edge in cycle_edges:
                edge_cycle_count[edge] += 1
        # Increment the count for each edge by 1 after checking all cycles
        edge_cycle_count[edge] += 1
    return edge_cycle_count

def calculate_node_strengths(G):
    out_strength, in_strength = {}, {}
    for node in G.nodes:
        out_strength[node] = sum(G[node][succ]['weight'] for succ in G.successors(node))
        in_strength[node] = sum(G[pred][node]['weight'] for pred in G.predecessors(node))
    return in_strength, out_strength

def bracket(node, in_strength, out_strength):
    return in_strength[node] + out_strength[node]

def calculate_importance(G, edge_cycle_count, in_strength, out_strength):
    importance_scores = {}
    for node in G.nodes():
        product = 0
        for x in list(G.neighbors(node)):
            edge = (node, x) if (node, x) in G.edges else (x, node)
            cycle_count = edge_cycle_count[edge] if edge in edge_cycle_count else 1
            factor = bracket(node, in_strength, out_strength) + bracket(x, in_strength, out_strength) - 2 * G.get_edge_data(*edge)['weight']
            w = G.get_edge_data(*edge)['weight']
            omjer = bracket(node, in_strength, out_strength) / (bracket(node, in_strength, out_strength) + bracket(x, in_strength, out_strength))
            partial_umnozak = (cycle_count-1) * factor * w * omjer
            print((node, x), cycle_count)
            product += partial_umnozak

        # Now, calculate the degrees sum and adjust the total umnozak
        strength_sum = in_strength[node] + out_strength[node]
        adjusted_umnozak = product + strength_sum
        
        importance_scores[node] = adjusted_umnozak  # Store the adjusted umnozak for normalization
        
        # Print the total umnozak and adjusted umnozak for each node
        print(f"Total umnozak for Node {node}: {product}")
        print(f"Total umnozak for Node {node} + in-degree + out-degree: {adjusted_umnozak}\n")

    # Normalize the scores
    suma = sum(importance_scores.values())
    normalized_scores = {node: score / suma * 100 for node, score in importance_scores.items()}
    return normalized_scores


def main():    
    G = nx.DiGraph()
    weighted_edges = [
        (1, 2, 3), 
        (1, 4, 2.9), (4, 1, 0.1),
        (1, 5, 1.9), (5, 1, 0.1), 
        (1, 6, 2.4),  (6, 1, 0.1),
        (1, 7, 0.3), (7, 1, 0.1),
        (2, 1, 0.8),
        (2, 20, 0.7), (20, 2, 0.1), 
        (2, 22, 0.6), (22, 2, 0.1), 
        (3, 1, 1.65), (1, 3, 0.1),
        (3, 31, 0.9), (31, 1, 0.1),
        (3, 40, 0.4), (40, 3, 0.1),
        (4, 3, 1.3), (3, 4, 0.1), 
        (4, 41, 0.65), (41, 4, 0.1),
        (4, 2, 0.3), (2, 4, 0.1), 
        (5, 52, 0.9), (52, 5, 0.1), 
        (5, 51, 0.7), (51, 5, 0.1), 
        (5, 50, 0.3),  (50, 5, 0.1),
        (6, 5, 1.4), (5, 6, 0.1), 
        (6, 20, 1.4), (20, 6, 0.1), 
        (6, 61, 0.4), (61, 6, 0.1), 
        (6, 60, 1.1), (60, 6, 0.1),
        (7, 72, 0.3), (72, 7, 0.1), 
        (7, 70, 1.9), (70, 7, 0.1), 
        (21, 2, 0.7), (2, 21, 0.1),
        (23, 2, 0.9), (2, 23, 0.1),
        (32, 3, 0.3), (3, 32, 0.1), 
        (30, 3, 0.5), (3, 30, 0.1), 
        (40, 4, 1.4), (4, 40, 0.1), 
        (42, 4, 0.3), (4, 42, 0.1), 
        (52, 6, 0.4), (6, 52, 0.1), 
        (60, 2, 1.4), (2, 60, 0.1), 
        (71, 7, 0.7), (7, 71, 0.1),
        (7, 73, 1), 
        (73, 7, 1), 
        (73, 732, 0.4), (732, 73, 0.1), 
        (731, 73, 0.4), (73, 731, 0.1),
        (733, 73, 0.4), (73, 733, 0.1)
    ]
    G.add_weighted_edges_from(weighted_edges)
    
    edge_cycle_count = count_cycles_containing_edges(G)
    in_strength, out_strength = calculate_node_strengths(G)
    importance_scores = calculate_importance(G, edge_cycle_count, in_strength, out_strength)

    for node, score in importance_scores.items():
        print(f"Node {node} importance score: {score} %")   

# Execute the main function
main()








