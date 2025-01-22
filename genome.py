"""
Main genome file for generating a genotype that can be used for generating a neural net
"""

import random
import numpy as np
import activations


class NeuronGene:
    """Node that represents a neuron"""
    def __init__(self, neuron_id:int, bias:float, activation_func) -> None:
        self.neuron_id = neuron_id
        self.bias = bias
        self.activation_func = activation_func
        self.current_value = random.uniform(-1,1)
        #apparently holds a list of nodes that can be connected?

class LinkId:
    """looks at two nodes and makes a pointer from input to output"""
    def __init__(self,input_id:int, output_id:int) -> None:
        self.input_id = input_id #NeuronGene ids passed in
        self.output_id= output_id

class LinkGene:
    """Contains the pointers and the weight that is multiplied from linkID input"""
    def __init__(self, link_id:LinkId, weight:float, is_enabled:bool) -> None:
        self.link_id = link_id
        self.weight = weight
        self.is_enabled = is_enabled

class Genome:
    """Contains Everything needed for a NN(phenotype)"""
    def __init__(self, genome_id:int, num_inputs:int, num_outputs:int) -> None:
        self.genome_id = genome_id#default genome has no hidden layers and is fully connected as of now
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.neurons = [] #contains NeuronGene nodes
        self.links = []#contains LinkGenes
        self.fill_link_neurons()

    def _random_bias(self):#silly function but helps readability
        bias = random.uniform(-1,1)
        return bias

    def _random_weight(self):
        weight = random.uniform(-1,1)
        return weight

    def fill_link_neurons(self):#cant be used for adding and removing Nodes CHANGE NAME!!!!!
        """This does not feel very efficient but as long as the NNs stay small should be good"""
        current_id = 0
        dict_of_input_nodes = {}
        dict_of_output_nodes = {}
        dict_of_hidden_nodes = {}
        for node in range(self.num_inputs):#generates the input nodes
            dict_of_input_nodes[current_id] = NeuronGene(current_id, self._random_bias(), activations.Relu())
            current_id += 1

        for node in range(self.num_outputs):#generates the output nodes
            dict_of_output_nodes[current_id] = NeuronGene(current_id, self._random_bias(), activations.Softmax())#TBD will be softmax for now
            current_id += 1

        #for node in dict_of_input_nodes.items():#makes the links for a dense NN
        #    for out in dict_of_output_nodes.items():
        #        link = LinkId(node[1].neuron_id, out[1].neuron_id)
        #        self.links.append(LinkGene(link, self._random_weight(), True))   ###ONLY FOR DENSE NNs

        self.neurons= {"input":dict_of_input_nodes,
                       "output":dict_of_output_nodes, 
                       "hidden":dict_of_hidden_nodes}#not sure I love having nodes organized like this

    def forward_pass(self):
        """looks at a link then indexes each node to do calculations for a forward pass"""
        start_node, end_node, current_start_node = None, None, None
        for link in self.links:
            start_node = self.neurons["input"][link.link_id.input_id]

            if start_node is not current_start_node:#stops start node bias from being continually added in
                start_node.current_value = start_node.activation_func.forward(start_node.current_value + start_node.bias)
                current_start_node = start_node
            #ugly solution

            end_node = self.neurons["output"][link.link_id.output_id]
            end_node.current_value += start_node.current_value * link.weight

    def add_link(self):
        """Creates a random new link"""
        node1 = random.choice(list(self.neurons['input']))
        node2 = random.choice(list(self.neurons['output']))

        self.links.append(LinkGene(LinkId(node1, node2), self._random_weight(),True))


if __name__ == "__main__":
    t_genome = Genome(genome_id= 1, num_inputs=6, num_outputs=1)

    print("'Inputs'")
    for neuron in t_genome.neurons["input"]:
        print(t_genome.neurons["input"][neuron].current_value)
    print("-----------------------")
    print("'Outputs'")
    for neuron in t_genome.neurons["output"]:
        print(t_genome.neurons["output"][neuron].current_value)

    t_genome.add_link()


    t_genome.forward_pass()
    print("\n")

    print("'Inputs'")
    for neuron in t_genome.neurons["input"]:
        print(t_genome.neurons["input"][neuron].current_value)
    print("-----------------------")
    print("'Outputs'")
    for neuron in t_genome.neurons["output"]:
        print(t_genome.neurons["output"][neuron].current_value)

"""
TODO:
    start brainstorming how to add in nodes and play with having mutations
    "kinda complete"maybe separate the nodes into 3 dictionaries of "input, hidden and output" (in def forward_pass)
    apply softmax to output nodes
    add hidden nodes in theory to forward pass
    Make it so add link cant add if the link already exists
"""