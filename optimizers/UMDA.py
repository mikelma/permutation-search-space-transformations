import numpy as np
import permu_utils as putils
import math
import datetime

class TimeoutError(Exception):
    def __init__(self, message):
        super().__init__(message)

class UMDA():

    def __init__(self):
        pass

    def learn_distribution(self, pop, shape, dtype=np.int32):
        '''Learn probability distibution based on the given population matrix.
        
        Args:
            pop (ndarray): Population matrix of permutations of size n.
            shape (tupe): (n, m) the shape of the probability matrix, nxm.
            dtype (numpy data type): Type of the integers in the probability matrix. 
                               Default: np.int32.

        Returns:
            ndarray: nxm matrix. 
        '''
        # n = pop.shape[1]
        n, m = shape
        freq = np.empty(shape, dtype=dtype)        
        pop = np.hsplit(pop, m)

        for i in range(n):
            for j in range(m):
                freq[i][j] = np.count_nonzero(pop[j] == i)

        return freq

    # def sample_population_ad_hoc(self, p, n, 
    #                       pop=np.array([]),
    #                       timeout=None, 
    #                       quit_in_timeout=False):
    #     '''Given a probability matrix of size mxm, sample n permutations
    #     of length m.

    #     Args: 
    #         p (ndarray): probability matrix.
    #         n (int): number of permutations to sample.
    #         timeout (int or None): Default: None. Enable timeouti, in milliseconds.  
    #         quit_in_timeout (bool): Default: False. Exit the program if timeout occurs.
    #                                 Else, stop sampling and return obtained results.
    #     
    #     Returns:
    #         ndarray: matrix with the sampled permutations

    #     '''
    #     samples = set()
    #     # print('p: ', p) # debug
    #     size = p.shape[0] # Size of permutations
    #     identity = np.array(range(size))

    #     assert n <= math.factorial(size), 'Number of permutations to sample is too large.'

    #     start = datetime.datetime.now()

    #     while len(samples) < n:
    #         delta_t = datetime.datetime.now() - start
    #         if type(timeout) == int and int(delta_t.total_seconds() * 1000) >= timeout:
    #             print('Warning: Timeout passed when sampling permutations.')
    #             if quit_in_timeout:
    #                 quit() 
    #             else:
    #                 return None
    #                 # break
    #         # Generate permu
    #         permu = []
    #         for j in range(size): # For each position
    #             # Probability for elements in the j's position 
    #             p_ = np.delete(p[j], permu, axis=0) 
    #             available = np.delete(identity, permu) 
    #             try:
    #                 if sum(p_) == 0: 
    #                     rand = 0 
    #                 else:
    #                     rand = np.random.uniform(0, sum(p_))
    #             except Exception as e:
    #                 print(e, '\n', 'Program state before error:')
    #                 print('p_: ', p_, ' available: ', available, '  rand: ', rand) #debug
    #                 quit()
    #             i = 0
    #             s = 0 # sum
    #             while s < rand:
    #                 s += p_[i] 
    #                 if s < rand:
    #                     i += 1
    #             permu.append(available[i])
    #         i = 0 
    #         repeated = False
    #         while not repeated and i < pop.shape[0]: 
    #             repeated = tuple(pop[i]) == permu
    #             i += 1
    #         if not repeated: 
    #             samples.add(tuple(permu))

    #     return  putils.set2np(samples, size) 

    # @profile
    def sample_population(self, p, n_samples, 
                          permutation,
                          pop=np.array([]),
                          timeout=None):
        '''Given a probability matrix of size nxm, n_samples number of solutions 
        of length m.

        Args: 
            p (ndarray): probability matrix.
            n_samples (int): number of samples.
            permutation (bool): If true, samples are going to be permutations.
                                Default: True.
            pop (ndarray): Individuals from pop matrix are not going to be 
                           repeated in the sampled matrix. Warning, the larger 
                           the size of pop, sampling will be more time-consuming. 
                           Default: empty array. 
            timeout (int or None): Enable timeouti, in milliseconds. Default: None.
        
        Returns:
            ndarray: sampled matrix. 

        '''
        samples = set()
        # print('p: ', p) # debug
        size = p.shape[0] # Size of solutions to sample 
        if not permutation:
            size -= 1
        identity = np.array(range(p.shape[0]))

        # p = p.T

        start = datetime.datetime.now()

        while len(samples) < n_samples:
            ## Watch for timeouts
            delta_t = datetime.datetime.now() - start
            if type(timeout) == int and int(delta_t.total_seconds() * 1000) >= timeout:
                raise TimeoutError('Error: Timeout passed when sampling new solutions.')

            # Generate permu
            permu = []
            for j in range(size): # For each position
                # Probability for elements in the j's position 
                if permutation:
                    p_ = np.delete(p[j], permu, axis=0) 
                    available = np.delete(identity, permu) 
                else:
                    p_ = p[j]

                if sum(p_) == 0: 
                    rand = 0 
                else:
                    rand = np.random.uniform(0, sum(p_))

                i = 0
                s = 0 # sum
                while s < rand:
                    s += p_[i] 
                    if s < rand:
                        i += 1

                if permutation:
                    permu.append(available[i])
                else:
                    permu.append(identity[i])
            
            i = 0 
            repeated = False
            while not repeated and i < pop.shape[0]: 
                repeated = tuple(pop[i]) == permu
                i += 1
            if not repeated: 
                samples.add(tuple(permu))

        return  putils.set2np(samples, size) 
