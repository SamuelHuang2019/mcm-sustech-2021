import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

G.add_node(1)
G.add_nodes_from([2, 3])

G.add_nodes_from([
    (4, {"color": "red"}),
    (5, {"color": "green"}),
])

H = nx.path_graph(10)
G.add_nodes_from(H)

G.add_node(H)

G.add_edge(1, 2)
e = (2, 3)
G.add_edge(*e)  # unpack edge tuple*

G.add_edges_from([(1, 2), (1, 3)])
G.add_edges_from(H.edges)

G.clear()

G.add_edges_from([(1, 2), (1, 3)])
G.add_node(1)
G.add_edge(1, 2)
G.add_node("spam")  # adds node "spam"
G.add_nodes_from("spam")  # adds 4 nodes: 's', 'p', 'a', 'm'
G.add_edge(3, 'm')

print(G.number_of_nodes())

print(G.edges([2, 'm']))
print(G.nodes)
print(G.edges)

print(G.edges([2, 'm']))
plt.plot()
nx.draw(G, with_labels=True, font_weight='bold')
plt.show()

H = nx.petersen_graph()
plt.plot()
nx.draw(H, with_labels=True, font_weight='bold')
plt.show()
