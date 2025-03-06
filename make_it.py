import gymnasium as gym
import visualize


def run(env, pop, generations_to_run):
    """runs a population for generations_to_run in a given environment
    returns the most fit individual, makes a file logging the generation and highest fitness"""
    open("most_recent_run_gens.txt", "w", encoding="utf-8").close# used to wipe file clean after each run
    #file mostly used for bug testing
    current_generation = 0

    while current_generation != generations_to_run:
        for individual in pop.pop:
            
            individual.fitness = 0#resets individual's fitness each gen
            observation, info = env.reset()#resets the environment for each individuals attempt
            episode_over = False

            while not episode_over:
                action = individual.gen_move(observation) #generates a action from observations
                observation, reward, terminated, truncated, info = env.step(action)#takes a action
                individual.fitness += reward #increments reward. this is specific to cartpole env

                episode_over = terminated or truncated

        pop.fitness_sort()#sorts pop by fitness
        pop.mutate()#mutates population
        print("best fitness =", pop.pop[0].fitness)
        current_generation += 1
        print(current_generation)

        with open("Most_recent_run_Gens.txt", "a", encoding="utf-8") as file:
                #for bug testing
                file.write(f"CURRENT GEN:{current_generation}\n")
                file.write(f"fitness ={pop.pop[0].fitness}\n")
                file.write("\n")
                file.close()

    env.close()
    return pop.pop[0]#returns most fit individual

def play_game(individual, game):
    """Only works for cart pole. displays the individual thats passed in attempting cartpole
    shows the neural network after the attempt"""
    env = gym.make(game, render_mode = "human")
    individual.fitness = 0
    observation, info = env.reset()
    episode_over = False

    while not episode_over:
        action = individual.gen_move(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        individual.fitness += reward

        episode_over = terminated or truncated
    visualize.display_NN(individual.neural_net)
    print("done")
    env.close()