import environment
import population


env = environment.Simple_Navigation_Env(8,8)

pop = population.Population(1000,env,4,5)

pop.run(200,30)


"""
Things that could be causing a problem:
the environment- I like this one the best
how mutation is handled-
individuals dont share anything- no mutation off of the top performers
not enough attempts- start messing with multi processing
reset current value of all nodes to 0 after each forward pass
"""

"""time to start making graphs to see if things are actually learning"""
"""somehow fitness is going up but no links are being added"""
"""Adjust how mutations happen and their rates"""
"""use the top fitness as the ones to mutate off of in new generations?"""
