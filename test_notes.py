import environment
import population
import multiprocessing
import multiprocessing
import itertools

def worker_function(arg1, arg2, arg3):
    """Function to be executed by each process."""
    return f"Process with {arg1}, {arg2}, {arg3}"

if __name__ == '__main__':
    rotating_args = itertools.cycle(['A', 'B', 'C'])
    static_args1 = 10
    static_args2 = 20
    num_processes = 5
    
    argument_tuples = [(next(rotating_args), static_args1, static_args2) for _ in range(10)]
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(worker_function, argument_tuples)
    
    for result in results:
        print(result)

"""
Things that could be causing a problem:
the environment- I like this one the best
how mutation is handled-
individuals dont share anything- no mutation off of the top performers
not enough attempts- start messing with multi processing
fix add_link in genome so it can have a link be added with the input being from a hidden node
maybe add a chance to remove hidden nodes
SET SEEDS SO YOU CAN TELL WHAT HELPS
"""

"""time to start making graphs to see if things are actually learning"""
"""Adjust how mutations happen and their rates"""
"""use the top fitness as the ones to mutate off of in new generations?"""
