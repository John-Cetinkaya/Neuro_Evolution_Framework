from genome import Genome
import matplotlib.pyplot as plt
import networkx as nx

test_g = Genome(0,4,1)
test_g.add_link()
test_g.add_node()
test_g.add_link()
test_g.add_link()
test_g.add_node()
test_g.add_node()
test_g.add_node()
test_g.add_node()
test_g.add_node()


def display_NN(net):
    G = nx.DiGraph()  # Or use nx.DiGraph for directed graph

    for node_type in net.neurons:
        for node in net.neurons[node_type].values():
            G.add_node(node.neuron_id)
    for link in net.links:
        G.add_edge(link.link_id.input_id, link.link_id.output_id, weight= link.weight)

    pos = {}  # Dictionary to store node positions

    layer_index = 0
    tnode = 0
    for layer in net.neurons:

        for node in net.neurons[layer]:

            pos[node] = (layer_index, tnode)  # Adjust 'layer_index' for visual separation
            tnode += 1
        layer_index += 1
        tnode = 0

    nx.draw(G, pos=pos, with_labels=True, node_color='b', edge_color='grey')

    plt.show()
