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

    # @profile
    def sample_population(self, p, 
                          n_samples, 
                          permutation,
                          pop,
                          fitness,
                          eval_func,
                          check_repeat,
                          timeout=None,
                          dtype=np.int8):
        '''Given a probability matrix of size nxm, n_samples number of solutions 
        of length m.

        Args: 
            p (ndarray): probability matrix.
            n_samples (int): number of samples.  permutation (bool): If true, samples are going to be permutations.
                                Default: True.
            permutation (bool): Set true if the solutions to sample are permutations, else False.
            pop (ndarray): Individuals from pop matrix are not going to be 
                           repeated in the sampled matrix. 
            fitness (ndarray): Array of the fitness values of the population given. Its shape must be (1, pop_size).
            eval_func: Instance of the evaluation function.
            timeout (int or None): Enable timeout, in milliseconds. Default: None.
            dtype (numpy type): Data type of the sampled solutions. Default: np.int8
        
        Returns:
            tuple(ndarray, ndarray) : sampled solutions matrix and the fitness array of the sampled solutions. 

        '''

        size = p.shape[0] # Size of solutions to sample 

        if not permutation:
            size -= 1

        identity = np.array(range(p.shape[0]))

        samples = np.empty((n_samples, size), dtype=dtype)
        samples_f = np.empty(n_samples)

        start = datetime.datetime.now()
        
        n_sampled = 0 # Number of permutations sampled and added to the new pop 
        while n_sampled < n_samples:
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
            
            # Evaluate the sampled permu
            f = eval_func(permu)

            if f not in fitness and check_repeat:
                i = 0
                repeated = False
                while not repeated and i < pop.shape[0]:
                    repeated = np.all(pop[i] == permu)
                    i += 1

                if not repeated:
                    samples[n_sampled] = permu
                    samples_f[n_sampled] = f
                    n_sampled += 1

            elif not check_repeat:
                samples[n_sampled] = permu
                samples_f[n_sampled] = f
                n_sampled += 1
                 
        return samples, samples_f 
