import environment
import population


env = environment.Simple_Navigation_Env(40,40)

pop = population.Population(500,env,2,4)

pop.run(2000,80)



"""time to start making graphs to see if things are actually learning"""
"""somehow fitness is going up but no links are being added"""
"""Adjust how mutations happen and their rates"""
"""use the top preformers as the ones to mutate off of in new generations?"""
