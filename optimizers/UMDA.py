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

    def sample_population(self, 
                          p, 
                          samples,
                          samples_f,
                          pop,
                          pop_f,
                          eval_func,
                          check_repeat,
                          timeout=None):
        '''Given a probability matrix of size nxm, n_samples number of solutions 
        of length m.

        Args: 
            p (ndarray): probability matrix.
            samples (ndarray): Matrix where samples are going to be stored.
            samples_f (ndarray): Array where the fitness values of the sampled solutions are going to be stored.
            pop (ndarray): Individuals from pop matrix are not going to be 
                           repeated in the sampled matrix. 
            pop_f (ndarray): Fitness array of the given population (pop).
            eval_func: Instance of the evaluation function.
            check_repeat (bool): Check if the sampled solution exists in the population, solutions won't be repeated.. 
            timeout (int or None): Enable timeout, in milliseconds. Default: None.
        
        Returns:
            tuple(ndarray, ndarray) : sampled solutions matrix and the fitness array of the sampled solutions. 
        '''
        size = min(p.shape) # Size of the permutation to sample 
        permutation = p.shape[0] == p.shape[1] # Define search space

        start = datetime.datetime.now()
        n_sampled = 0 # Number of permutations sampled and added to the new pop 

        while n_sampled < samples.shape[0]:

            ## Watch for timeouts
            delta_t = datetime.datetime.now() - start
            if type(timeout) == int and int(delta_t.total_seconds() * 1000) >= timeout:
                raise TimeoutError('Error: Timeout passed when sampling new solutions.')

            # Generate sample
            sample = []
            for j in range(size): # For each position

                # Probability for elements in the j's position 
                p_ = p[j]

                if permutation:
                    s_max = 0
                    for i in range(size): 
                        if i not in sample:
                            s_max += p_[i]
                else:
                    s_max = sum(p[0])

                rand = np.random.uniform(0, s_max)
                ##############################################
                if rand != 0:

                    s = 0
                    i = 0

                    while s < rand:

                        if permutation and i not in sample:
                            s += p_[i]

                        elif not permutation:
                            s += p_[i]

                        if s < rand:
                            i += 1
                else:
                    # print('debuug'*10)
                    i = 0
                    while i in sample:
                        i += 1
                
                sample.append(i)
                ##############################################

            # If Vjs, transform vj to permu
            if not permutation:
                sample = putils.vj2permu(np.array(sample))


            # Evaluate the sampled permu
            f = eval_func(sample)

            if f in pop_f and check_repeat:
                # Check if the sampled solution exists in the population
                i = 0
                repeated = False
                while not repeated and i < pop.shape[0]:
                    repeated = np.all(pop[i] == sample)
                    i += 1

                if not repeated:
                    # Add the sampled solution to the population 
                    samples[n_sampled] = sample
                    samples_f[n_sampled] = f
                    n_sampled += 1

            # elif not check_repeat:
            else:
                # Do not check if the sampled ppulation already exists in pop
                samples[n_sampled] = sample
                samples_f[n_sampled] = f
                n_sampled += 1

        return samples, samples_f 

    def sample_population_v2(self, 
                          p, 
                          samples,
                          samples_f,
                          pop,
                          pop_f,
                          eval_func,
                          permutation,
                          check_repeat,
                          timeout=None):
        '''New sampling method.
        '''
        pass

