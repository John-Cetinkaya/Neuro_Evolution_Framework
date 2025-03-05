import multiprocessing
import gymnasium as gym
import make_it
import population

env = gym.make('CartPole-v1')

pop = population.Population(200, env, 4, 2)


def multi_process_prep(population, game):
    prepped_data = []
    for individual in population:
        prepped_data.append((individual, gym.make(game)))
    return prepped_data

def test_multi_run(pop):

    pop.fitness = 0#resets individual's fitness each gen
    observation, info = pop.env.reset()#resets the environment for each individuals attempt
    episode_over = False

    while not episode_over:
        action = pop.gen_move(observation) #generates a action from observations
        observation, reward, terminated, truncated, info = pop.env.step(action)#takes a action
        pop.fitness += reward #increments reward. this is specific to cartpole env

        episode_over = terminated or truncated
    



if __name__ == '__main__':
    data = multi_process_prep(pop.pop, 'CartPole-v1')

    current_generation = 0
    generations_to_run = 40
    while current_generation != generations_to_run:
        with multiprocessing.Pool(3) as pool:
            pool.map(test_multi_run, data)

        pop.fitness_sort()#sorts pop by fitness
        pop.mutate()#mutates population
        print("best fitness =", data[0].fitness)
        current_generation += 1
        print(current_generation)
  