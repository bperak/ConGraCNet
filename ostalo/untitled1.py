python3 import networkx as nx

G = nx.Graph()

G.add_edges_from([(1, 2), (1, 3)])

nx.draw(G)
