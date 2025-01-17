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
        self.genome_id = genome_id#make default genome have no hidden layers
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.neurons = [] #contains NeuronGene nodes
        #generated at the beginning
        self.links = []#contains LinkGenes
        self.fill_link_neurons()

    def _random_bias(self):#silly function but helps readability
        bias = random.uniform(-1,1)
        return bias

    def _random_weight(self):
        weight = random.uniform(-1,1)
        return weight

    def fill_link_neurons(self):#cant be used for adding and removing Nodes
        """This does not feel very efficient but as long as the NNs stay small should be good"""
        current_id = 0
        list_of_output_nodes = []
        list_of_input_nodes = []
        for node in range(self.num_inputs):#generates the input nodes
            list_of_input_nodes.append(NeuronGene(current_id, self._random_bias(), activations.Relu()))
            current_id += 1

        for node in range(self.num_outputs):#generates the output nodes
            list_of_output_nodes.append(NeuronGene(current_id, self._random_bias(), activations.Softmax()))# might need 2 lists to start TBD will be softmax for now
            current_id += 1

        for node in list_of_input_nodes:#makes the pointers for a dense NN
            for out in list_of_output_nodes:
                link = LinkId(node.neuron_id, out.neuron_id)
                self.links.append(LinkGene(link, self._random_weight(), True))

        self.neurons= list_of_input_nodes + list_of_output_nodes

    def forward_pass(self):
        """looks at each link then locates the start of the link and the end once it is found does the calculations
        to find what the end of the links new value should be"""
        start_node, end_node = None, None
        for link in self.links:
            for node in self.neurons:
                if link.link_id.input_id == node.neuron_id:
                    start_node = node
                elif link.link_id.output_id == node.neuron_id:
                    end_node = node
                elif start_node and end_node is not None:
                    break
            start_node.current_value = start_node.activation_func.forward(start_node.current_value + start_node.bias)
            end_node.current_value = start_node.current_value * link.weight
            #final activation of softmax is not applied
            #STORE NEURONS IN A DICT FOR FASTER FORWARD PASSES?
            #SET UP FOR TESTING NEXT AND COMMENT


if __name__ == "__main__":
    t_genome = Genome(genome_id= 1, num_inputs=3, num_outputs=3)
    print(1)
    for neurons in t_genome.neurons:
        print(neurons.current_value)
    t_genome.forward_pass()
    print("\n")
    for neurons in t_genome.neurons:
        print(neurons.current_value)
    print(1)