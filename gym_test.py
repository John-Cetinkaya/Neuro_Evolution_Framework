import gymnasium as gym
import population

env = gym.make('CartPole-v1')

observation, info = env.reset()

pop = population.Population(1000, env, 4, 2)

current_generation = 0
max_generation = 100

while current_generation != max_generation:
    for individual in pop.pop:
        
        individual.fitness = 0
        observation, info = env.reset()
        episode_over = False
        while not episode_over:
            action = individual.gen_move(observation)  # agent policy that uses the observation and info
            observation, reward, terminated, truncated, info = env.step(action)
            individual.fitness += reward

            episode_over = terminated or truncated

    pop.fitness_sort()
    pop.mutate()
    print("best fitness =", pop.pop[0].fitness)
    current_generation += 1
    print(current_generation)


    if current_generation == max_generation:
        env.close()
        env = gym.make('CartPole-v1', render_mode = "human")

        best_of_best = pop.pop[0]
        best_of_best.fitness = 0
        observation, info = env.reset()
        episode_over = False
        while not episode_over:
            action = best_of_best.gen_move(observation)  # agent policy that uses the observation and info
            observation, reward, terminated, truncated, info = env.step(action)
            best_of_best.fitness += reward

            episode_over = terminated or truncated
        print("done")
env.close()