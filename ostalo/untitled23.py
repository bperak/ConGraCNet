import networkx as nx
from matplotlib import pyplot as plt

def calculate_node_importance(G):
    # First, calculate the weighted degree for each node
    weighted_degree = {node: sum(data['weight'] for _, _, data in G.edges(node, data=True)) for node in G.nodes}
    
    # Initialize a dictionary to store the importance scores
    importance_scores = {}
    
    # Iterate through each node to calculate its importance
    for node in G.nodes:
        # Initialize the sum of importance contributions from connected edges
        sum_importance = 0
        for neighbor in G.neighbors(node):
            edge_data = G.get_edge_data(node, neighbor)
            edge_weight = edge_data['weight']
            # Calculate z using the weighted degree of the node and its neighbor
            z = edge_weight * weighted_degree[node] / (weighted_degree[node] + weighted_degree[neighbor])
            # Assume lambda (cycle factor) as 1 for simplicity, adjust as needed based on cycle analysis
            lambda_factor = 1
            # Calculate the importance contribution from this edge
            edge_importance = lambda_factor * z
            # Add to the total sum of importance for the node
            sum_importance += edge_importance
        
        # Add the node's own weighted degree to its importance score
        importance_scores[node] = sum_importance + weighted_degree[node]
    
    # Normalize the importance scores
    total_importance = sum(importance_scores.values())
    normalized_scores = {node: score / total_importance * 100 for node, score in importance_scores.items()}
    return normalized_scores

def main():
    G = nx.Graph()
    
    #nx.add_cycle(G, [10,20,30])
    #nx.add_cycle(G, [3, 4, 26])
    nx.add_cycle(G, [1, 2, 4])
    nx.add_cycle(G, [1, 3, 4])
    nx.add_cycle(G, [1, 5, 6])
    nx.add_cycle(G, [3, 4, 40])
    #nx.add_cycle(G, [1, 2, 20, 6])
    nx.add_cycle(G, [5, 6, 52])
    #nx.add_cycle(G, [2, 60, 6, 20])
    #nx.add_cycle(G, [1, 3, 8, 6])

    #G.add_edge(1, 10, weight=0.4)
    G.add_edge(1, 2, weight=3)
    G.add_edge(1, 3, weight=1.75)
    G.add_edge(1, 4, weight=3)
    #G.add_edge(4, 10, weight=0.4)
    G.add_edge(1, 5, weight=2)
    G.add_edge(1, 6, weight=2.5)
    G.add_edge(1, 7, weight=0.4)
    G.add_edge(2, 4, weight=0.4)
    G.add_edge(2, 20, weight=0.8)
    G.add_edge(2, 21, weight=2)
    G.add_edge(2, 22, weight=0.7)
    G.add_edge(2, 23, weight=1)
    G.add_edge(2, 60, weight=1.5)
    #G.add_edge(3, 8, weight=0.4)
    G.add_edge(3, 4, weight=0.4)
    G.add_edge(3, 40, weight=0.5)
    #G.add_edge(4, 2, weight=0.4)
    G.add_edge(3, 30, weight=0.6)
    G.add_edge(3, 31, weight=1)
    G.add_edge(3, 32, weight=0.4)
    G.add_edge(4, 40, weight=1.5)
    G.add_edge(4, 41, weight=0.75)
    #G.add_edge(4, 42, weight=0.4)
    G.add_edge(4, 42, weight=0.4)
    #G.add_edge(4, 44, weight=0.4)
    #G.add_edge(4, 45, weight=0.4)
    #G.add_edge(6, 8, weight=0.4)
    G.add_edge(5, 6, weight=1.5)
    G.add_edge(5, 50, weight=0.4)
    G.add_edge(5, 51, weight=0.8)
    G.add_edge(5, 52, weight=1)
    #G.add_edge(5, 53, weight=0.4)
    G.add_edge(6, 52, weight=0.4)
    G.add_edge(6, 60, weight=1.2)
    G.add_edge(6, 61, weight=0.5)
    G.add_edge(7, 70, weight=2)
    G.add_edge(7, 71, weight=0.8)
    G.add_edge(7, 72, weight=0.4)
    #G.add_edge(8, 80, weight=0.4)

    
    # Calculate node importance
    importance_scores = calculate_node_importance(G)
    
    # Print node importance scores
    for node, score in sorted(importance_scores.items(), key=lambda item: item[1], reverse=True):
        print(f"Node {node} importance score: {score:.2f}")

    # Optional: Visualize the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

if __name__ == "__main__":
    main()
