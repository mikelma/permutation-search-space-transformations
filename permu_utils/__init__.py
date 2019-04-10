import numpy as np
import math
import random
import itertools as it

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

def discordancesToPermut(indCode, refer):
    n = len(indCode)
    rem = refer[:]  #[i for i in refer]
    sigma = np.zeros(n,dtype=np.int)
    for i in range(n):
        sigma[i] = rem[indCode[i]]
        rem.pop(indCode[i])
    return sigma#[i+1 for i in permut];

def kendallTau(A, B=None):
    if B is None : B = list(range(len(A)))
    pairs = it.combinations(range(0, len(A)), 2)
    distance = 0
    for x, y in pairs:
        #if not A[x]!=A[x] and not A[y]!=A[y]:#OJO no se check B
        a = A[x] - A[y]
    try:
        b = B[x] - B[y]# if discordant (different signs)
    except:
        print("ERROR kendallTau, check b",A, B, x, y)
    # print(b,a,b,A, B, x, y)
    if (a * b < 0):
        distance += 1
    return distance

def permu2vj(permu): 
    '''Transform a permutation its Vj representation.

    Args:
        permu (ndarray): Numpy permutation.

    Returns:
        ndarray: Vj array. 
    '''
    vj = []
    i = 0
    while len(permu) > 1:
        # vj.append((permu > permu[0]).sum())
        vj.append((permu < permu[0]).sum())
        permu = np.delete(permu, 0)
        i += 1
    return np.array(vj)

def vj2permu(vj):
    '''Transform a vector from Vj space to permutation space. 

    Args:
        vj (ndarray): Numpy array, Vj vector.

    Returns:
        ndarray: Permutation vector.
    '''
    # e = list(reversed(range(vj.shape[0]+1)))
    e = list(range(vj.shape[0]+1))
    permu = []
    for elem in vj:
        permu.append(e[elem])        
        del e[elem]
    permu.append(e[0])
    return np.array(permu)

def transform(pop, func):
    '''Applies a trasformation function to every individual of the population.
    
    Args:
        pop (ndarray): population matrix.
        func (function): instance of the transformation function.

    Returns:
        ndarray: Transformated population matrix.
    '''
    return np.array(list(map(func, pop)))


