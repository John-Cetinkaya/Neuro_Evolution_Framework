"""
For creating a visualization of a given Genome
"""

from genome import Genome
import networkx as nx
import matplotlib.pyplot as plt

test_g = Genome(0,2,1)
test_g.add_link()
test_g.add_link()
test_g.add_node()


G = nx.DiGraph()
for node_type in test_g.neurons:
    for node in test_g.neurons[node_type].values():
        G.add_node(node.neuron_id)
for link in test_g.links:
    G.add_edge(link.link_id.input_id, link.link_id.output_id, weight= link.weight)

#G.add_edge(1, 2)
#e = (2, 3)
#G.add_edge(*e)
#G.add_edge("Node 1", "Node 2")
#t =nx.topological_sort(G)
layers=test_g.neurons
x = nx.topological_sort(G)
pos = nx.multipartite_layout(x,subset_key=layers)
nx.draw(G,pos, with_labels=True)
plt.show()
print("hi")