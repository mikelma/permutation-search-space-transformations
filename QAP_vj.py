# Optimum: QAP: 122455319

import problems
from optimizers import UMDA 
import permu_utils as putils
import numpy as np
import matplotlib.pyplot as plt 

PERMU_LENGTH = 20
POP_SIZE = PERMU_LENGTH*20
SURV_RATE = .5
ITERS = 2 
TIMEOUT = 4*1000
INSTANCE_NAME = 'tai20b.dat'

permu_dtype = np.int8

# Initialize optimizer and problem
umda = UMDA()
qap = problems.QAP()

dist, flow = qap.load_instance(INSTANCE_NAME)

# Create population
pop = putils.random_population(PERMU_LENGTH,
                                    POP_SIZE)
n_surv = int(POP_SIZE*SURV_RATE) # Number of survivor solutions

# Init loggers
log_min = []
log_avg = []

# Evaluate for the initial population
fitness = np.empty(POP_SIZE)
for indx in range(POP_SIZE):
    fitness[indx] = qap.evaluate(pop[indx], dist, flow)

for iter_ in range(ITERS):

    # print('iter ', iter_+1, '/', ITERS, 
    #       'mean: ', np.mean(fitness), ' best: ', min(fitness))
    # log_min.append(min(fitness))
    # log_avg.append(np.mean(fitness))
    
    # For later use
    old_pop = pop
    old_f = fitness

    # Select best solutions
    surv = np.empty((n_surv, PERMU_LENGTH), dtype=permu_dtype)
    surv_f = np.empty(n_surv) # Fitness of survivors 
    for i in range(n_surv):
        bests_indx = np.argmin(fitness)
        surv[i] = pop[bests_indx]
        surv_f[i] = fitness[bests_indx]

        pop = np.delete(pop, bests_indx, axis=0)
        fitness = np.delete(fitness, bests_indx)
    
    ### Print and log data ###
    print('iter ', iter_+1, '/', ITERS, 
          'mean: ', np.mean(surv_f), ' best: ', min(surv_f))
    log_min.append(min(surv_f))
    log_avg.append(np.mean(surv_f))

    # Transform survivor permus to vj
    surv_vj = putils.transform(surv, putils.permu2vj)

    # Learn a probability distribution from survivors
    p = umda.learn_distribution(surv_vj, shape=(PERMU_LENGTH, PERMU_LENGTH-1))

    # Sample new solutions
    try:
        new_vj = umda.sample_population(p, n_surv, 
                                        permutation=False,
                                        pop=np.array([]), 
                                        timeout=TIMEOUT)
    except Exception as e:
        print(e)
        plt.plot(range(iter_+1), log_avg, label='Mean')
        plt.plot(range(iter_+1), log_min, label='Best')
        plt.title('Vj ' + INSTANCE_NAME 
                  + ' best: {:0.2f}'.format(min(log_min)))
        plt.xlabel('Iterations')
        plt.ylabel('Survivors fitness')
        plt.legend()
        plt.grid(True)
        plt.show()
        putils.fancy_matrix_plot(p, 'Last probability matrix')
        quit()

    # Transform population of vj to permus
    new = putils.transform(new_vj, putils.vj2permu)    

    # Evaluate the sampled solutions
    new_f = np.empty(n_surv) # Fitness of  
    for i in range(n_surv):
        new_f[i] = qap.evaluate(new[i], dist, flow)

    # fitness = np.hstack((surv_f, new_f)) 
    # pop = np.vstack((surv, new))  
    fitness = np.hstack((old_f, new_f))
    pop = np.vstack((old_pop, new))

    # Second selection
    pop, fitness = putils.remove_from_pop(pop, fitness, n_surv, func='max')

plt.plot(range(ITERS), log_avg, label='Mean')
plt.plot(range(ITERS), log_min, label='Best')
plt.xlabel('Iterations')
plt.ylabel('Survivors fitness')
plt.legend()
plt.title('Vj ' + INSTANCE_NAME 
          + ' best {:0.2f}'.format(min(log_min)))
plt.grid(True)
plt.show()

putils.fancy_matrix_plot(p, 'Last probability matrix')
