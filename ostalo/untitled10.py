import networkx as nx

def find_cycles_containing_edge(G, edge, cycles):
    """
    Find and count cycles that contain a given directed edge.
    
    Parameters:
    - G: A directed graph (nx.DiGraph)
    - edge: A tuple representing a directed edge (source, target)
    - cycles: A list of cycles, where each cycle is represented as a list of nodes
    
    Returns:
    - count: The number of cycles containing the edge
    - cycles_with_edge: A list of cycles that contain the edge
    """
    # Convert each cycle into a list of directed edges for easy comparison
    cycle_edges = [list(zip(cycle, cycle[1:] + [cycle[0]])) for cycle in cycles]
    
    # Initialize a list to hold cycles that include the given directed edge
    cycles_with_edge = []
    
    # Iterate through the cycles to check if the directed edge is present
    for cycle, edges in zip(cycles, cycle_edges):
        if edge in edges:  # Check for the presence of the directed edge
            cycles_with_edge.append(cycle)
    
    # Count the number of cycles containing the edge
    count = len(cycles_with_edge)
    
    return count, cycles_with_edge

def main():
    # Create a directed graph object
    G = nx.DiGraph() 
    
    # Define weighted edges to add to the graph
    weighted_edges = [
        (1, 2, 3), (1, 4, 3), (1, 5, 2), (1, 6, 2.5), (1, 7, 0.4), (2, 1, 0.8), (2, 20, 0.8),
        (2, 22, 0.7), (3, 1, 1.75), (3, 31, 1), (3, 40, 0.5), (4, 3, 1.4), (4, 41, 0.75),
        (4, 2, 0.4), (5, 52, 1), (5, 51, 0.8), (5, 50, 0.4), (6, 5, 1.5), (6, 20, 1.5),
        (6, 61, 0.5), (6, 60, 1.2), (7, 72, 0.4), (7, 70, 2), (21, 2, 0.8), (23, 2, 1),
        (32, 3, 0.4), (30, 3, 0.6), (40, 4, 1.5), (42, 4, 0.4), (52, 6, 0.4), (60, 2, 1.5),
        (71, 7, 0.8), (7, 73, 1), (73, 7, 1), (73, 732, 0.5), (731, 73, 0.5), (733, 73, 0.5)
    ]
    G.add_weighted_edges_from(weighted_edges) 

    # Get simple cycles in the graph
    cycles = list(nx.simple_cycles(G))

    # Example: Check if a specific directed edge is part of any cycle and how many
    #edge_to_check = (2, 1)  # Change this edge as needed
    edge_to_check = (1, 2) 
    count, cycles_with_edge = find_cycles_containing_edge(G, edge_to_check, cycles)
    
    print(f"Directed edge {edge_to_check} is part of {count} cycle(s).")

# Execute the main function
main()
