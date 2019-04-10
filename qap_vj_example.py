import problems
import optimizers
import permu_utils as putils
import numpy as np

PERMU_LENGTH = 12
POP_SIZE = PERMU_LENGTH*100
SURV_RATE = .5
ITERS = 100
TIMEOUT = 10*1000

print('Problem dimensionality: ', PERMU_LENGTH)
print('Population size: ', POP_SIZE)

umda = optimizers.UMDA()
qap = problems.QAP(PERMU_LENGTH)

# qap.generate_instance('qap'+str(PERMU_LENGTH), 1, 10, 0, 1)
# quit()

dist, flow = qap.load_instance('qap'+str(PERMU_LENGTH))

# Create population
pop = putils.random_population(PERMU_LENGTH,
                                    POP_SIZE)
fitness = np.array([None]*POP_SIZE) 

for iter_ in range(ITERS):

    # Evaluate 
    to_eval = np.where(fitness == None)[0]
    for indx in to_eval: 
        fitness[indx] = qap.evaluate(pop[indx], dist, flow)

    # Select best
    n_survs = int(POP_SIZE*SURV_RATE)
    surv = np.empty((n_survs, PERMU_LENGTH), dtype=np.int)
    surv_f = np.array([])
    for i in range(surv.shape[0]):
        bests_indx = np.argmin(fitness)
        surv[i] = pop[bests_indx]
        surv_f = np.hstack((surv_f, fitness[bests_indx]))

        pop = np.delete(pop, bests_indx, axis=0)
        fitness = np.delete(fitness, bests_indx)

    print('mean fitness: ', np.mean(surv_f), ' best: ', min(surv_f))

    # Transform survivor permus to vj
    surv_vj = putils.transform(surv, putils.permu2vj)

    # Learn probability matrix
    p = umda.learn_distribution(surv_vj)

    # Sample new solutions
    # new = umda.sample_population(p, n_survs, pop=pop, 
    #                              timeout=TIMEOUT, quit_in_timeout=True)
    new_vj = umda.sample_population(p, n_survs, pop=np.array([]), 
                                 timeout=TIMEOUT, quit_in_timeout=True)

    # Transform population of vj to permus
    new = putils.transform(new_vj, putils.vj2permu)    

    # Create the new population 
    fitness = np.hstack((surv_f, [None]*n_survs)) 
    pop = np.vstack((surv, new))  
    # print('fitness: ', fitness)

