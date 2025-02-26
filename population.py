import genome
import random
import numpy as np
from activations import Softmax


class Individual:
    """An individual that contains a Neural Net and a fitness to interact with a environment"""
    def __init__(self, genome_id, input_neurons, output_neurons, action_set):
        self.neural_net = genome.Genome(genome_id, input_neurons, output_neurons)
        self.fitness = 0
        self.actions = action_set #list of allowed moves should be same shape as outputs

    def __lt__(self, other):
        if self.fitness > other.fitness:
            return self.fitness > other.fitness

    def gen_move(self,inputs):#in example inputs are position
        """makes a prediction and makes a choice from the action set"""
        #assigns input values
        for node_index in range(len(self.neural_net.neurons["input"])):
            self.neural_net.neurons["input"][node_index].current_value = inputs[node_index]
        activation_test = Softmax()
        action = activation_test.forward(self.neural_net.forward_pass())#makes a choice by applying softmax
        return self.actions[np.argmax(action)]

    def set_fitness(self, fitness):
        self.fitness = fitness


class Population:#have population track fitness maybe store individuals and fitness in a tuple?
    """A population to do all of your shenanigans"""
    def __init__(self, size, env, input_neurons, output_neurons):
        #TESTING CODE
        open("most_recent_run_gens.txt", "w").close
        self.env = env
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.pop = self.gen_pop(size)

    def gen_pop(self,size):
        """generates a population"""
        pop = []
        genome_id = 0
        for _ in range(size):
            pop.append(Individual(genome_id, self.input_neurons, self.output_neurons, self.env.actions))
            genome_id += 1
        return pop

    def select_new_pop_tourney(self, k,percent_kept = .33):
        pop_to_keep = round(percent_kept * len(self.pop))
        new_pop = []
        new_pop.extend(self.pop[:pop_to_keep])
        for _ in self.pop[pop_to_keep:]:
            choices = random.sample(self.pop, k=k)
            choices.sort()
            new_pop.append(choices[0])
        self.pop = new_pop

    def mutate(self, percent_kept = .33):
        """mutates the population does not touch highest fitness critters"""
        pop_to_keep = round(percent_kept * len(self.pop))
        for individual in self.pop[pop_to_keep:]:
            individual.neural_net.mutate()

    def run(self, num_of_gens, ticks):#one individual tries at a time
        current_gen = 0
        while current_gen != num_of_gens:
            for individual in self.pop:
                for frame in range(ticks):
                    prediction = individual.gen_move(self.env.observations)
                    self.env.move(step = 1, prediction = prediction)
                individual.set_fitness(self.env.gen_fitness())
                self.env.reset()


            self.fitness_sort()
            self.mutate()

            print("fitness =",self.pop[0].fitness)
            print(current_gen)
            with open("Most_recent_run_Gens.txt", "a") as file:
                file.write(f"CURRENT GEN:{current_gen}\n")
                file.write(f"fitness ={self.pop[0].fitness}\n")
                file.write(f"fitness ={self.env.goal}\n")
                file.write("\n")
                file.close()
            current_gen += 1


    def fitness_sort(self):
        """sorts the individuals based on fitness"""
        self.pop.sort()







if __name__ == "__main__":


    test_guy = Individual(0,4,4, [1,-1,2,3])
    test_guy.neural_net.add_link()
    test_guy.neural_net.add_link()
    test_guy.neural_net.add_link()

    print(test_guy.gen_move([0,0,0,0]))

    test_pop = Population(4, None, 3, 2)

    for i in test_pop.pop:
        fit = random.randint(0,10)
        i.fitness = fit

    

    test_pop.fitness_sort()
    print (test_pop)