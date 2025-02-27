import gymnasium as gym
import population
import make_it

env = gym.make('CartPole-v1')

observation, info = env.reset()

pop = population.Population(200, env, 4, 2)

best_guy = make_it.run(env, pop, 100)

make_it.play_game(best_guy)