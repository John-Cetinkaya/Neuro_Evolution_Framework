"""Only contains display_NN"""

import matplotlib.pyplot as plt
import networkx as nx


def display_NN(net):
    """displays a given neural network from an Individual doesnt work perfect yet"""
    G = nx.DiGraph()

    for node_type in net.neurons:
        for node in net.neurons[node_type].values():
            G.add_node(node.neuron_id)
    for link in net.links:
        G.add_edge(link.link_id.input_id, link.link_id.output_id, weight= link.weight)

    pos = {}

    layer_index = 0
    pos_node = 0
    for layer in net.neurons:

        for node in net.neurons[layer]:

            pos[node] = (layer_index, pos_node)
            pos_node += 1
        layer_index += 1
        pos_node = 0

    nx.draw(G, pos=pos, with_labels=True, node_color='b', edge_color='grey')

    plt.show()
