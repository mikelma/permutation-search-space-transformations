import configparser
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
                 timeout,
                 check_repeat,
                 permu_dtype=np.int8):
        
        
        assert space in ['permutation', 'vj'], 'Please select a valid search space type.'

        self.space = space
        self.timeout = timeout
        self.check_repeat = check_repeat

        self.size = size
        self.pop_size = pop_size

        self.evaluate = evaluator
        
        self.n_surv = int(pop_size*surv_rate)

        self.iters = iters

        self.permu_dtype = permu_dtype

        self.umda = UMDA()

    # @profile
    def run(self):

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

            # Log data
            log['min'].append(np.min(pop_f))
            log['max'] .append(np.max(pop_f))
            log['mean'].append(np.mean(pop_f))
            log['median'].append(np.median(pop_f))

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

            if self.space == 'permutation':
                # Learn distribution
                p = self.umda.learn_distribution(surv,
                                            shape=(self.size, 
                                                   self.size)) 
            elif self.space == 'vj':
                # Transform survivor permus to vj
                surv_vj = putils.transform(surv, putils.permu2vj)

                # Learn distribution
                p = self.umda.learn_distribution(surv_vj,
                                            shape=(self.size,
                                                   self.size-1))

            p = p.T # NOTE: Temporary solution
            # NOTE: If permutation apply laplace, sum 1 to the probability matrix
            if self.space == 'permutation':
                p += 1
            
            # Sample new solutions
            # try:
            samples, samples_f = self.umda.sample_population_fast(p=p, 
                                                             samples=samples,
                                                             samples_f=samples_f,
                                                             pop=pop,
                                                             pop_f=pop_f,
                                                             eval_func=self.evaluate,
                                                             check_repeat=self.check_repeat,
                                                             timeout=self.timeout)
            # except Exception as e:
            #     print('[!] Timeout exception occurred. Returning log.')
            #     print(e)
            #     return log

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
        ############################################
        best = pop[np.argsort(pop_f)[0]]
        print('Best solution: ', best)
        if not putils.is_permutation(pop):
            print('Not permu found in pop!')
            quit()
        ############################################

        return log
                
if __name__ == '__main__':

    from problems import QAP
    from problems import PFSP
    import matplotlib.pyplot as plt
    
    INSTANCE_NAME = 'instances/QAP/tai20b.dat'
    # INSTANCE_NAME = 'instances/PFSP/tai20_5_0.fsp'
    SIZE = 20
    POP_SIZE = 200
    SURV_RATE = .5
    ITERS = 200
    # SPACE = 'permutation'
    SPACE = 'vj'
    TIMEOUT = 3*1000
    CHECK_REPEAT = True
    DTYPE = np.int8

    problem = QAP()
    # problem = PFSP()

    dist, flow = problem.load_instance(INSTANCE_NAME)
    # instance = problem.load_instance(INSTANCE_NAME)

    def evaluate(permu):
        return problem.evaluate(permu, dist, flow)

    # def evaluate(permu):
    #     return problem.evaluate(permu, instance)

    alg = Algorithm(size=SIZE,
                    pop_size=POP_SIZE,
                    evaluator=evaluate,
                    surv_rate=SURV_RATE,
                    iters=ITERS,
                    space=SPACE,
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

