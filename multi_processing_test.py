import multiprocessing
import gymnasium as gym
import population
import time
import itertools

def fitness_eval(individual):
    """Evaluates the fitness of a individual
    HAS TO REMAKE ENV EVERY TIME IT RUNS AND THAT MAKES ME SAD"""
    env = gym.make("Acrobot-v1")#this adds 80 seconds on a population of 500 for 100 generations
    #it scales with population size each env build is about .011 seconds tested on Acrobot-v1
    individual.fitness = 0#resets individual's fitness each gen
    observation, info = env.reset()#resets the environment for each individuals attempt
    episode_over = False

    while not episode_over:
        action = individual.gen_move(observation) #generates a action from observations
        observation, reward, terminated, truncated, info = env.step(action)#takes a action
        individual.fitness += reward #increments reward. this is specific to cartpole env

        episode_over = terminated or truncated

    return individual

def multi_core_run(num_cores, generations_to_run, population):
    """runs a population with a specified amount of cores"""
    current_generation = 0
    with multiprocessing.Pool(num_cores) as pool:
        while current_generation != generations_to_run:
            evaluated_pop = pool.map(fitness_eval, population.pop)
            population.pop = evaluated_pop

            population.fitness_sort()#sorts pop by fitness
            population.mutate()#mutates population

            print("best fitness =", population.pop[0].fitness)
            current_generation += 1
            print(current_generation)


if __name__ == '__main__':
    #env4 = gym.make('CartPole-v1')
    #env3 = gym.make('CartPole-v1')
    #env2 = gym.make('CartPole-v1')
   # env1 = gym.make('CartPole-v1')

    #envs = itertools.cycle([env1,env2,env3,env4])
    start_time = time.perf_counter()
    env = gym.make("Acrobot-v1")
    end_time = time.perf_counter()
    print(f"runtime = {end_time - start_time}")
    pop = population.Population(100, env, 6, 3)

    multi_core_run(6, 100, pop)
    end_time = time.perf_counter()

    print(f"runtime = {end_time - start_time}")
