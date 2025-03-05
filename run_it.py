import gymnasium as gym
import population
import make_it

#finds a solution faster if only links can be added not neurons
env = gym.make('CartPole-v1')

pop = population.Population(100, env, 4, 2)

best_guy = make_it.run(env, pop, 100)

make_it.play_game(best_guy,'CartPole-v1')



"""
env_Acrobat = gym.make("Acrobot-v1")

pop = population.Population(100, env_Acrobat, 5, 3)

best_guy = make_it.run(env_Acrobat, pop, 30)

make_it.play_game(best_guy,"Acrobot-v1")
"""


"""
#Doesnt work too well with this problem. assumption is issue with mutation
env_lunar = gym.make("LunarLander-v3")

observation, info = env_lunar.reset()

pop = population.Population(300, env_lunar, 4, 4)

best_guy = make_it.run(env_lunar, pop, 100)

make_it.play_game(best_guy,"LunarLander-v3")
"""