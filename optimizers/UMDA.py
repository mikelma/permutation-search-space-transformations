import numpy as np
import permu_utils as putils
import math
import datetime

class UMDA():

    def __init__(self):
        pass

    def learn_distribution(self, pop):
        '''Learn probability distibution based on the given population matrix.
        
        Args:
            pop (ndarray): Population matrix of permutations of size n.

        Returns:
            ndarray: nxn matrix. 
        '''
        n = pop.shape[1]
        freq = np.empty((n, n), dtype=np.int)        
        pop = np.hsplit(pop, n)

        for i in range(n):
            for j in range(n):
                freq[i][j] = np.count_nonzero(pop[j] == i)

        return freq

    @profile
    def sample_population(self, p, n, 
                          pop=np.array([]),
                          timeout=None, 
                          quit_in_timeout=False):
        '''Given a probability matrix of size mxm, sample n permutations
        of length m.

        Args: 
            p (ndarray): probability matrix.
            n (int): number of permutations to sample.
            timeout (int or None): Default: None. Enable timeouti, in milliseconds.  
            quit_in_timeout (bool): Default: False. Exit the program if timeout occurs.
                                    Else, stop sampling and return obtained results.
        
        Returns:
            ndarray: matrix with the sampled permutations

        '''
        samples = set()
        # print('p: ', p) # debug
        size = p.shape[0] # Size of permutations
        identity = np.array(range(size))

        assert n <= math.factorial(size), 'Number of permutations to sample is too large.'

        start = datetime.datetime.now()

        while len(samples) < n:
            delta_t = datetime.datetime.now() - start
            if type(timeout) == int and int(delta_t.total_seconds() * 1000) >= timeout:
                print('Warning: Timeout passed when sampling permutations.')
                if quit_in_timeout:
                    quit() 
                else:
                    break
            # Generate permu
            permu = []
            for j in range(size): # For each position
                # Probability for elements in the j's position 
                p_ = np.delete(p[j], permu, axis=0) 
                available = np.delete(identity, permu) 
                try:
                    if sum(p_) == 0: 
                        rand = 0 
                    else:
                        rand = np.random.uniform(0, sum(p_))
                except Exception as e:
                    print(e, '\n', 'Program state before error:')
                    print('p_: ', p_, ' available: ', available, '  rand: ', rand) #debug
                    quit()
                i = 0
                s = 0 # sum
                while s < rand:
                    s += p_[i] 
                    if s < rand:
                        i += 1
                permu.append(available[i])
            i = 0 
            repeated = False
            while not repeated and i < pop.shape[0]: 
                repeated = tuple(pop[i]) == permu
                i += 1
            if not repeated: 
                samples.add(tuple(permu))

        return  putils.set2np(samples, size) 

