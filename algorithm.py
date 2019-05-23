import numpy as np
import permu_utils as putils

from optimizers import UMDA

class Algorithm():
    
    def __init__(self,
                 size, 
                 pop_size,
                 evaluator,
                 surv_rate,
                 iters,
                 space,
                 sampling_func,
                 timeout,
                 check_repeat,
                 permu_dtype=np.int8):
        '''Algortithm constructor.
            
        Args:
            size (int): Size of the problem, length of the permutations.
            pop_size (int): Number of individuals in the population.
            evaluator (func): Evaluation function, is given a permutation and 
                              returns a float of the fitness value.
            surv_rate (float): number of survivor solutions = pop_size*surv_rate.
            iters (int): Number of iterations.
            space (str): Search space. (Ex.: 'permutation', 'vj').
            timeout (int): Number of milliseconds until timeout error sampling.
            check_repeat (bool): If true, sampled solutions will be solutions 
                                 that do not exist in the population.
            permu_dtype (numpy dtype): Numpy dtype permutations. Ex.: np.int8.

        Returns:
            Algorithm instance.
        '''
        self.check_repeat = check_repeat
        self.timeout = timeout
        self.size = size
        self.pop_size = pop_size
        self.evaluate = evaluator
        self.n_surv = int(pop_size*surv_rate)
        self.iters = iters
        self.permu_dtype = permu_dtype

        self.umda = UMDA()

        self.sampling_func = sampling_func

        # Define search space specific variables
        if space == 'permutation':
            self.transform = False
            self.space2permu = None 

        elif space == 'vj':
            self.transform= True
            self.permu2space = putils.permu2vj
            self.space2permu = putils.vj2permu

        else:                                  
            print('Please select a valid search space type.')
            quit()

    # @profile
    def run(self, verbose=True):
        '''Runs the algorithm with the given parameters in the constructor.
        Args:
            verbose (bool): If true, prints mean, min and max fitness of the
                            current population in the command line in each
                            iteration.
        Returns:
            log (dict): a dictionary with 'min', 'max', 'mean' and 'median' keys,
                        storing data of each generation.
        '''
        # Init loggers
        log = {'min':[],
               'max':[],
               'mean':[],
               'median':[]}
    
        # Sample initial random population
        pop = putils.random_population(self.size,
                                       self.pop_size,
                                       self.permu_dtype)
        # Initialize fitness array 
        pop_f = np.empty(self.pop_size)

        # Initialize survivor's matrix and their fitness array
        surv = np.empty((self.n_surv, self.size), dtype=self.permu_dtype)
        surv_f = np.empty(self.n_surv)

        # Initialize samples matrix and fitness array
        samples = np.empty((self.n_surv, 
                            self.size),
                            dtype=self.permu_dtype)
        samples_f = np.empty(self.n_surv)
        
        # Evaluate initial population
        for i in range(self.pop_size):
            pop_f[i] = self.evaluate(pop[i])

        ### MAIN LOOP ###

        for iter_ in range(self.iters):

            ############################################
            #   NOTE: DEBUG
            ############################################
            # if not putils.is_permutation(pop):
            #     best = pop[np.argsort(pop_f)[0]]
            #     print('Not permu found in pop!')
            #     print('Best solution: ', best)
            #     print('space2permu: ', self.space2permu)
            #     print('Sampling func: ', self.sampling_func)
            #     print('transformation: ', self.transform)
            #     quit()
            # else:
            #     print('--> Pop Ok!')
            ############################################

            # Log data
            log['min'].append(np.min(pop_f))
            log['max'] .append(np.max(pop_f))
            log['mean'].append(np.mean(pop_f))
            log['median'].append(np.median(pop_f))

            if verbose:
                print('iter ', iter_+1, '/', 
                      self.iters, 
                      ' mean: ', log['mean'][-1],
                      ' best: ', log['min'][-1])

            # Order from min to max fitness values, index list
            ranking = np.argsort(pop_f)

            # Select best solutions
            ranking = ranking[:self.n_surv]
            
            for i, indx in enumerate(ranking):
                surv[i] = pop[indx]
                surv_f[i] = pop_f[indx]


            if self.transform:
                # Transform survivors
                surv_transformed = putils.transform(surv, self.permu2space)

                p = self.umda.learn_distribution(surv_transformed, self.size)
            else:
                p = self.umda.learn_distribution(surv, self.size)

            # putils.fancy_matrix_plot(p, title=str(p.shape))
            
            # Sample new solutions
            samples, samples_f = self.umda.sample_population_v2(p=p, 
                                                             sampling_func=self.sampling_func,
                                                             samples=samples,
                                                             samples_f=samples_f,
                                                             pop=pop,
                                                             pop_f=pop_f,
                                                             eval_func=self.evaluate,
                                                             transformation=self.space2permu,
                                                             check_repeat=self.check_repeat,
                                                             timeout=self.timeout)

            # Ranking, the best fitness valued solutions index
            ranking = np.argsort(pop_f)

            ranking_samples = np.argsort(samples_f)
            
            # Replace the best new solutions with the 
            # worst solutions from the population

            indexes = list(reversed(range(self.pop_size)))

            stop = False
            i = 0
            
            while i < self.n_surv and not stop:

                worst_pop = ranking[indexes[i]]
                best_sample = ranking_samples[i]

                if samples_f[best_sample] <= pop_f[worst_pop]:

                    pop[worst_pop] = samples[best_sample]
                    pop_f[worst_pop] = samples_f[best_sample]

                else:
                    stop = True

                i += 1

        return log
                
if __name__ == '__main__':

    from problems import QAP
    from problems import PFSP
    import matplotlib.pyplot as plt
    from optimizers import UMDA

    umda = UMDA()
    
    SIZE = 20
    POP_SIZE = 200
    SURV_RATE = .5
    ITERS = 700
    TIMEOUT = 3*1000
    CHECK_REPEAT = True
    DTYPE = np.int8

    # SPACE = 'permutation'
    # SAMPLING = umda.sample_ad_hoc_laplace 

    SAMPLING = umda.sample_no_restriction
    SPACE = 'vj'

    # INSTANCE_NAME = 'instances/QAP/tai20b.dat'
    # problem = QAP()
    # dist, flow = problem.load_instance(INSTANCE_NAME)
    # def evaluate(permu):
    #     return problem.evaluate(permu, dist, flow)
    
    INSTANCE_NAME = 'instances/PFSP/tai20_5_0.fsp'
    problem = PFSP()
    instance = problem.load_instance(INSTANCE_NAME)
    def evaluate(permu):
        return problem.evaluate(permu, instance)

    alg = Algorithm(size=SIZE,
                    pop_size=POP_SIZE,
                    evaluator=evaluate,
                    surv_rate=SURV_RATE,
                    iters=ITERS,
                    space=SPACE,
                    sampling_func=SAMPLING,
                    timeout=TIMEOUT,
                    check_repeat=CHECK_REPEAT,
                    permu_dtype=DTYPE)

    log = alg.run()
    
    iters = range(len(log['min']))

    plt.plot(iters, log['min'], label='min')
    plt.plot(iters, log['max'], label='max')
    plt.plot(iters, log['median'], label='median')
    plt.legend()

    plt.show()

