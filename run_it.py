import gymnasium as gym
import population
import make_it
"""
env = gym.make('CartPole-v1')

observation, info = env.reset()

pop = population.Population(200, env, 4, 2)

best_guy = make_it.run(env, pop, 100)

make_it.play_game(best_guy,'CartPole-v1')"""


"""
env_Acrobat = gym.make("Acrobot-v1")

observation, info = env_Acrobat.reset()

pop = population.Population(100, env_Acrobat, 5, 3)

best_guy = make_it.run(env_Acrobat, pop, 30)

make_it.play_game(best_guy,"Acrobot-v1")"""

env_lunar = gym.make("LunarLander-v3")

observation, info = env_lunar.reset()

pop = population.Population(300, env_lunar, 4, 4)

best_guy = make_it.run(env_lunar, pop, 100)

make_it.play_game(best_guy,"LunarLander-v3")