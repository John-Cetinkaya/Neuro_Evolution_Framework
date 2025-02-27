"""
Main genome file for generating a genotype that can be used for generating a neural net
"""
import random
import activations


class NeuronGene:
    """Node that represents a neuron"""
    def __init__(self, neuron_id:int, bias:float, activation_func) -> None:
        self.neuron_id = neuron_id
        self.bias = bias
        self.activation_func = activation_func
        self.current_value = 0#SHOULD THIS BE 0 AT every pass??

class LinkId:
    """looks at two nodes and makes a pointer from input to output"""
    def __init__(self,input_id:int, output_id:int) -> None:
        self.input_id = input_id #NeuronGene ids passed in
        self.output_id= output_id

    def __eq__(self, other):
        return (self.input_id == other.input_id and self.output_id == other.output_id)

class LinkGene:
    """Contains the pointers and the weight that is multiplied from linkID input"""
    def __init__(self, link_id:LinkId, weight:float, is_enabled:bool) -> None:
        self.link_id = link_id
        self.weight = weight
        self.is_enabled = is_enabled

    def __eq__(self, other):
        return self.link_id == other.link_id

class Genome:
    """Contains Everything needed for a NN(phenotype)"""
    def __init__(self, genome_id:int, num_inputs:int, num_outputs:int) -> None:
        self.genome_id = genome_id#default genome has no hidden layers and is fully connected as of now
        self.current_node_id = 0
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

    def fill_link_neurons(self):
        """This does not feel very efficient but as long as the NNs stay small should be good"""
        dict_of_input_nodes = {}
        dict_of_output_nodes = {}
        dict_of_hidden_nodes = {}
        for _ in range(self.num_inputs):#generates the input nodes MAKE INPUT FUC IDENTITY
            dict_of_input_nodes[self.current_node_id] = NeuronGene(self.current_node_id, self._random_bias(), activations.Identity())
            self.current_node_id += 1
        for _ in range(self.num_outputs):#generates the output nodes
            dict_of_output_nodes[self.current_node_id] = NeuronGene(self.current_node_id, self._random_bias(), activations.Softmax())#TBD will be softmax for now
            self.current_node_id += 1

        #for node in dict_of_input_nodes.items():#makes the links for a dense NN
        #    for out in dict_of_output_nodes.items():
        #        link = LinkId(node[1].neuron_id, out[1].neuron_id)
        #        self.links.append(LinkGene(link, self._random_weight(), True))   ###ONLY FOR DENSE NNs

        self.neurons= {"input":dict_of_input_nodes, 
                       "hidden":dict_of_hidden_nodes,
                       "output":dict_of_output_nodes}#not sure I love having nodes organized like this


    def reset_NN_values(self):
        """Sets all node values back to 0 after each forward pass"""
        for node_type in self.neurons:
            for node in self.neurons[node_type]:
                self.neurons[node_type][node].current_value = 0

    def forward_pass(self):
        """preforms a single pass over the Neural Network returns output nodes"""
        #maybe could set current values to 0 as you pass forward
        working_node = None
        results = []

        for link in self.links:
            input_node = link.link_id.input_id
            output_node = link.link_id.output_id

            for node_type in self.neurons:
                #adds in bias to first node in link 
                if input_node in self.neurons[node_type]:
                    first_node = self.neurons[node_type][input_node]
                    if working_node != first_node:#needed so bias isnt continuously added in if there are more than 1 edges for a node
                        working_node = first_node
                        working_node.current_value = working_node.activation_func.forward(working_node.current_value + working_node.bias)

                if output_node in self.neurons[node_type]:#multiply the weight
                    second_node = self.neurons[node_type][output_node]
                    second_node.current_value = working_node.current_value * link.weight

        #pulls just the results not a very pretty way of doing this
        for node in self.neurons["output"]:
            results.append(self.neurons["output"][node].current_value)
        
        self.reset_NN_values()
        return results

    def add_link(self): #will have to make work with hidden nodes too
        """Creates a random new link and checks if the link already exists"""
        node1 = random.choice(list(self.neurons['input']))
        node2 = random.choice(list(self.neurons['output']))
        new_link_id = LinkId(node1, node2)
        new_link = LinkGene(new_link_id, self._random_weight(),True)

        if new_link not in self.links:
            self.links.append(LinkGene(new_link_id, self._random_weight(),True))
            self.links = self._top_sort()

    def add_node(self):
        """Adds a node on and edge and adds a edge from the incoming to itself and from itself to outgoing
        also sets the weights of the edges so they should have the same output as well as has no added in bias to prevent 
        large changes"""
        try:
            link = random.choice(self.links)
            link.is_enabled = False
            self.neurons["hidden"][self.current_node_id] = NeuronGene(self.current_node_id, 0,activations.tanh())

            new_link_id = LinkId(link.link_id.input_id, self.current_node_id)
            self.links.append(LinkGene(new_link_id, 1, True))

            new_link_id = LinkId(self.current_node_id, link.link_id.output_id)
            self.links.append(LinkGene(new_link_id, link.weight, True))
            self.current_node_id += 1

            self.links.remove(link)
            self.links = self._top_sort()
        except:
            pass#pretty cheese fix

    def adjust_weight(self, upper= .5, lower= -.5):
        """adjusts a random link weight"""
        try:
            link = self.links[random.randrange(0,len(self.links))]
            link.weight += random.uniform(upper, lower)
        except:
            pass
    
    def adjust_bias(self, upper= .5, lower= -.5):
        node_type = random.choice(list(self.neurons.keys()))
        try:
            node_id = random.choice(list(self.neurons[node_type].keys()))
            node = self.neurons[node_type][node_id]
            node.bias += random.uniform(upper, lower)
        except:
            pass

    def _top_sort(self):
        node_edge_dict = {}
        sorted_nodes = []
        added_nodes = []
        sorted_links = []

        #generates all the nodes and gives default of 0 edges
        for node_type in self.neurons:
            for node in self.neurons[node_type]:
                node_edge_dict[node] = 0

        #fills in how many dependant nodes each node has
        for edge in self.links:
            node_edge_dict[edge.link_id.output_id] += 1

        while len(node_edge_dict) > 0:
            added_nodes = []#used to remove values from dictionary

            #checks if nodes are dependant on other nodes
            for node, edges in node_edge_dict.items():
                if edges == 0:
                    sorted_nodes.append(node)
                    added_nodes.append(node)

            #checks if the degree should be reduced for all values
            for link in self.links:
                if (link.link_id.input_id in sorted_nodes) and (link.link_id.input_id in node_edge_dict.keys()):
                    node_edge_dict[link.link_id.output_id] -= 1
                    sorted_links.append(link)

            #decrements the node if it is sorted
            for node in added_nodes:
                del node_edge_dict[node]

        return sorted_links

    def mutate(self, add_node_chance = .05, add_link_chance = .2, adjust_weight_chance = .5, adjust_bias_chance = .05, do_nothing = .2):
        """Mutates a network based on the weights passed in. Total of prams should equal 1 but doesnt need to"""
        options = ["add_node", "add_link", "adjust_weight", "adjust_bias", "do_nothing"]#maybe add remove node
        weights = [add_node_chance, add_link_chance, adjust_weight_chance, adjust_bias_chance, do_nothing]
        
        choice = random.choices(options, weights, k=1)[0]

        if choice == "add_node":
            self.add_node()

        if choice == "add_link":
            self.add_link()

        if choice == "adjust_weight":
            self.adjust_weight()

        if choice == "adjust_bias":
            self.adjust_bias()

        if choice == "do_nothing":
            pass



if __name__ == "__main__":
    t_genome = Genome(genome_id= 1, num_inputs=6, num_outputs=2)

    print("'Inputs'")
    for neuron in t_genome.neurons["input"]:
        print(t_genome.neurons["input"][neuron].current_value + random.uniform(0,40))
    print("-----------------------")
    print("'Outputs'")
    for neuron in t_genome.neurons["output"]:
        print(t_genome.neurons["output"][neuron].current_value)

    t_genome.add_link()
    t_genome.add_link()
    t_genome.add_link()
    t_genome.add_link()



    print(t_genome.forward_pass())
    print("\n")

    print("'Inputs'")
    for neuron in t_genome.neurons["input"]:
        print(t_genome.neurons["input"][neuron].current_value)
    print("-----------------------")
    print("'Outputs'")
    for neuron in t_genome.neurons["output"]:
        print(t_genome.neurons["output"][neuron].current_value)
