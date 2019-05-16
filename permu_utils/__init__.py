import numpy as np
import math
import random
import itertools as it
import matplotlib.pyplot as plt 

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

def is_permutation(pop):
    '''Checks if every solution of the given population is a permutation.

        Args:
            pop (ndarray) : Population matrix.
        
        Returns: 
            bool: True if every solution of the given population is a permutation.
    '''
    is_permu = True
    i = 0
    while is_permu and i < pop.shape[0]:

        permu = pop[i]
        e = 0
        while is_permu and e < permu.shape[0]:
            is_permu = len(np.where(permu == e)[0]) == 1
            # print('e: ', e, ' ', len(np.where(permu == e)[0]))
            e += 1
        i += 1

        if not is_permu:
            print(permu)

    return is_permu

def set2np(set_, n, dtype):
    '''Python set type to numpy ndarray matrix.

    Args:
        set_ (set): set to convert.
        n (int): lenght of the permutations inside set_.
        dtype: numpy type.

    Returns:
        ndarray
    '''
    return np.array(list(set_), dtype=dtype).reshape((len(set_), n))

def random_population(n, size, dtype):
    '''Generate a random population of permutations.

    Args:
        n (int) : length of the permutations. 
        size (int): size of the population.
        dtype: numpy type.

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
    return set2np(pset, n, dtype=dtype)

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
        pop (ndarray): Population matrix.
        func (function): Instance of the transformation function.

    Returns:
        ndarray: Transformated population matrix.
    '''
    return np.array(list(map(func, pop)))

def remove_from_pop(pop, fitness, n_del, func='min'):
    '''Deletes a given number of items from the pop and fitness arrays. 
    The deletion is done removing the lowest or highest fitness valued items.

    Args:
        pop (ndarray): Population matrix. Matrix must be 2D.
        fitness (ndarray): Fitness array. Array's shape must be (1, n)
        n_del (int): Number of items to remove.
        func (str): Delete maximum or minimum fitness valued items. 
                    Values must be 'min' or 'max'. 
    Returns:
        (ndarray, ndarray): First array is the new population matrix, 
                            the second is new fitness array.

    Raises:
        ValueError: If the value of func is neither 'min' nor 'max'. 

    '''
    if func == 'min':
        f = np.argmin
    elif func == 'max':
        f = np.argmax
    else:
        raise ValueError("func parameter must be 'min' or 'max'."
                         + " Please give a valid parameter value.") 

    for i in range(n_del):

        indx = f(fitness)
        
        fitness = np.delete(fitness, indx, 0)
        pop = np.delete(pop, indx, 0)
    
    return pop, fitness

def fancy_matrix_plot(m, title=None):
    '''Plots a heatmap of the given matrix.
    
    Args:
        m (ndarray): two dimentional numpy array.
        title (str or None): Title of the graphic. Default: None. 
    '''
    plt.matshow(m)
    # Loop over data dimensions and create text annotations.
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
           plt.text(j, i, np.around(m[i, j], 1),
                     ha="center", va="center", color="w")

    if type(title) == str:
        plt.title(title)

    plt.show()

def invert(sigma, dtype):
    '''Calculate the inverse of the given permutation.
    Args:
        sigma (array): The permutation to invert.
        dtype : Type of the numpy array the function returns.
    
    Returns:
        ndarray: Inverse of the given sigma permutation.
    '''
    inv = np.empty(len(sigma), dtype=dtype)
    for i in range(len(sigma)):
        inv[sigma[i]] = i

    return inv
