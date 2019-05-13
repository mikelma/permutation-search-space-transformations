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
        
        # Vj samples matrix
        if self.space == 'vj':
            samples_vj = np.empty((self.n_surv,
                                   self.size-1),
                                   dtype=self.permu_dtype)

        # Evaluate initial population
        for i in range(self.pop_size):
            pop_f[i] = self.evaluate(pop[i])

        # Order from min to max fitness values, index list
        ranking = np.argsort(pop_f)

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

            # Select best solutions
            ranking = ranking[:self.n_surv]
            
            for i, indx in enumerate(ranking):
                surv[i] = pop[indx]
                surv_f[i] = pop_f[indx]

            # print('Survs: \n', surv)
            # print('\nSurvs fitness: ', surv_f)

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
            
            # Sample new solutions
            if self.space == 'permutation':
                try:
                    samples, samples_f = self.umda.sample_population(p, 
                                                                     samples,
                                                                     samples_f,
                                                                     pop,
                                                                     pop_f,
                                                                     self.evaluate,
                                                                     permutation=True,
                                                                     check_repeat=self.check_repeat,
                                                                     timeout=self.timeout)
                except:
                    print('[!] Timeout exception occurred. Returning log.')
                    return log

            else:
                try:
                    samples_vj, samples_f = self.umda.sample_population(p, 
                                                                        samples_vj,
                                                                        samples_f,
                                                                        pop,
                                                                        pop_f,
                                                                        self.evaluate,
                                                                        permutation=False,
                                                                        check_repeat=self.check_repeat,
                                                                        timeout=self.timeout)
                except:
                    print('[!] Timeout exception occurred. Returning log.')
                    return log


                # Transform sampled vj to permus
                samples = putils.transform(samples_vj, putils.vj2permu)    

            # Ranking, the best fitness valued solutions index
            ranking = np.argsort(pop_f)

            ranking_samples = np.argsort(samples_f)
            
            # Replace the best new solutions with the 
            # worst solutions from the population

            indexes = list(reversed(range(self.pop_size)))

            # print('indx: ', indexes)
            # quit()

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
    import matplotlib.pyplot as plt
    
    INSTANCE_NAME = 'instances/QAP/tai20b.dat'
    SIZE = 20
    POP_SIZE = 200
    SURV_RATE = .5
    ITERS = 100
    # SPACE = 'permutation'
    SPACE = 'vj'
    TIMEOUT = 3*1000
    CHECK_REPEAT = True
    DTYPE = np.int8

    qap = QAP()

    dist, flow = qap.load_instance(INSTANCE_NAME)

    def evaluate(permu):
        return qap.evaluate(permu, dist, flow)

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

