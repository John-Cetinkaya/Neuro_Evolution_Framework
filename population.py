import genome

class Individual:
    def __init__(self, genome_id, input_neurons, output_neurons, action_set):
        self.neural_net = genome.Genome(genome_id, input_neurons, output_neurons)
        self.fitness = 0
        self.actions = action_set #list of allowed moves should be same shape as outputs
    
    def gen_move(self,inputs):
        for node_index in range(len(self.neural_net.neurons["inputs"])):
            self.neural_net.neurons["inputs"][node_index].current_value = inputs[node_index]
        action = self.neural_net.forward_pass()



class Population:#have population track fitness maybe store individuals and fitness in a tuple?
    """A population to do all of your shenanigans"""
    def __init__(self, size, env, input_neurons, output_neurons):
        self.pop = self.gen_pop(size)
        self.env = env
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
    
    def gen_pop(self,size):
        """generates a population"""
        pop = []
        genome_id = 0
        for _ in range(size):
            pop.append(Individual(genome_id, self.input_neurons, self.output_neurons, self.env.actions))
            genome_id += 1
        return pop
    
    def mutate(self):#fix to save the top 1/3 of performers
        for individual in self.pop:
            individual.neural_net.mutate()
    
    def step(self):
        for individual in self.pop:
            pass
