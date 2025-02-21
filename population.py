import genome
import random
import numpy as np
from activations import Softmax


class Individual:
    def __init__(self, genome_id, input_neurons, output_neurons, action_set):
        self.neural_net = genome.Genome(genome_id, input_neurons, output_neurons)
        self.fitness = 0
        self.actions = action_set #list of allowed moves should be same shape as outputs
    
    def __eq__(self, other):
        if self.fitness > other.fitness:
            return self.fitness > other.fitness
        if self.fitness < other.fitness:
            return self.fitness < other.fitness
    
    def gen_move(self,inputs):
        #assigns input values
        for node_index in range(len(self.neural_net.neurons["input"])):
            self.neural_net.neurons["input"][node_index].current_value = inputs[node_index]
        activation_test = Softmax()
        action = activation_test.forward(self.neural_net.forward_pass())#makes a choice by applying softmax
        return self.actions[np.argmax(action)]



class Population:#have population track fitness maybe store individuals and fitness in a tuple?
    """A population to do all of your shenanigans"""
    def __init__(self, size, env, input_neurons, output_neurons):
        self.env = env
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.pop = self.gen_pop(size)

    def gen_pop(self,size):
        """generates a population"""
        pop = []
        genome_id = 0
        for _ in range(size):
            pop.append(Individual(genome_id, self.input_neurons, self.output_neurons, [0,0]))#self.env.actions))
            genome_id += 1
        return pop

    def mutate(self):#fix to save the top 1/3 of performers
        for individual in self.pop:
            individual.neural_net.mutate()

    def step(self):
        for individual in self.pop:
            pass
    
    def fitness_sort(self):
        self.pop.sort()







if __name__ == "__main__":
    test_guy = Individual(0,3,2, [0,0])
    test_guy.neural_net.add_link()
    test_guy.neural_net.add_link()
    test_guy.neural_net.add_link()

    test_guy.gen_move([0,0,0])

    test_pop = Population(4, None, 3, 2)

    for i in test_pop.pop:
        fit = random.randint(0,10)
        i.fitness = fit

    

    test_pop.fitness_sort()
    print (test_pop)