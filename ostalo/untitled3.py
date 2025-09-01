import networkx as nx

G = nx.Graph()

nx.add_cycle(G, [1,2,3,4])
nx.add_cycle(G, [10,20,30])

e = (1, 10)

G.add_edge(1, 10)

nx.draw(G, with_labels = True)

in_path = lambda e, path: (e[0], e[1]) in path or (e[1], e[0]) in path
cycle_to_path = lambda path: list(zip(path+path[:1], path[1:] + path[:1]))
in_a_cycle = lambda e, cycle: in_path(e, cycle_to_path(cycle))
in_any_cycle = lambda e, g: any(in_a_cycle(e, c) for c in nx.cycle_basis(g))

for edge in G.edges():
    if in_any_cycle(edge, G)==True:
        print(edge)