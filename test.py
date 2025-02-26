import environment
import population


env = environment.Simple_Navigation_Env(8,8)

pop = population.Population(5,env,4,5)

pop.run(100,30)


"""
Things that could be causing a problem:
the environment-
how mutation is handled-
individuals dont share anything- no mutation off of the top performers
not enough attempts- start messing with multi processing
"""

"""time to start making graphs to see if things are actually learning"""
"""somehow fitness is going up but no links are being added"""
"""Adjust how mutations happen and their rates"""
"""use the top fitness as the ones to mutate off of in new generations?"""
