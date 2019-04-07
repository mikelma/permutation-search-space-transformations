import numpy as np
import math
import random

# def random_permutation(n):
#     '''Generate a random permutation of the specified length.
# 
#     Args:
#         n (int): size of the permutation.
#     
#     Returns: 
#         tuple: Permutation of length n. 
#     '''
#     permu = set()
#     while len(permu) < n:
#         permu.add(random.randint(0, n-1))
# 
#     return permu

def set2np(set_, n):
    '''Python set type to numpy ndarray matrix.

    Args:
        set_ (set): set to convert.
        n (int): lenght of the permutations inside set_.
    Returns:
        ndarray
    '''
    return np.array(list(set_)).reshape((len(set_), n))

def random_population(n, size):
    '''Generate a random population of permutations.

    Args:
        n (int) : length of the permutations. 
        size (int): size of the population.
    Returns:
        ndarray: Poulation of random permutations.
    '''
    # Check given  population size
    assert size <= math.factorial(n), 'Population size too large.' 
    
    l = list(range(n))
    pset = set()
    while len(pset) < size:
        random.shuffle(l)
        pset.add(tuple(l))

    # return np.array(list(pset)).reshape((size, n))
    return set2np(pset, n)

