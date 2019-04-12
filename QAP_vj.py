import problems
import optimizers
import permu_utils as putils
import numpy as np
import matplotlib.pyplot as plt 

def debug(m, f):
    for i in range(len(f)):
        print(m[i], '  ', f[i])

# PERMU_LENGTH = 5
PERMU_LENGTH = 9
# POP_SIZE = PERMU_LENGTH*10
POP_SIZE = PERMU_LENGTH*100
SURV_RATE = .5
ITERS = 70 
TIMEOUT = 10*1000
INSTANCE_NAME = 'qap9_01'
LR = .1

umda = optimizers.UMDA()
qap = problems.QAP(PERMU_LENGTH)

dist, flow = qap.load_instance(INSTANCE_NAME)

# Create population
pop = putils.random_population(PERMU_LENGTH,
                                    POP_SIZE)
n_surv = int(POP_SIZE*SURV_RATE) # Number of survivor solutions

# Initialization of the prob. distribution 
p_ = np.zeros((PERMU_LENGTH-1, PERMU_LENGTH-1)) 

# Evaluate for the initial population
fitness = np.empty(POP_SIZE)
for indx in range(POP_SIZE):
    fitness[indx] = qap.evaluate(pop[indx], dist, flow)

log_min = []
log_avg = []

for iter_ in range(ITERS):

    print('mean fitness: ', np.mean(fitness), ' best: ', min(fitness))
    log_min.append(min(fitness))
    log_avg.append(np.mean(fitness))
    
    # Select best solutions
    surv = np.empty((n_surv, PERMU_LENGTH), dtype=np.int)
    surv_f = np.empty(n_surv) # Fitness of survivors 
    for i in range(n_surv):
        bests_indx = np.argmin(fitness)
        surv[i] = pop[bests_indx]
        surv_f[i] = fitness[bests_indx]

        pop = np.delete(pop, bests_indx, axis=0)
        fitness = np.delete(fitness, bests_indx)
    
    # print('surv:', '\n', surv)

    # Transform survivor permus to vj
    surv_vj = putils.transform(surv, putils.permu2vj)
    # print('surv vj:','\n', surv_vj)

    # Learn a probability distribution from survivors
    p = umda.learn_distribution(surv_vj)

    if LR != None:
        # Apply learning rate
        p = LR*p_ + (1-LR)*p
        p_ = p # Pi-1 = Pi
    
    # print('p: ', '\n', p)

    # Sample new solutions
    # new_vj = umda.sample_population(p, n_surv, pop=np.array([]), 
    #                              timeout=TIMEOUT, quit_in_timeout=True)
    new_vj = umda.sample_population(p, n_surv, pop=surv_vj, 
                                 timeout=TIMEOUT, quit_in_timeout=True)
    # print('new vj: ', '\n', new_vj)

    # Transform population of vj to permus
    new = putils.transform(new_vj, putils.vj2permu)    
    # print('new: ', '\n', new)

    # Evaluate the sampled solutions
    new_f = np.empty(n_surv) # Fitness of  
    for i in range(n_surv):
        new_f[i] = qap.evaluate(new[i], dist, flow)

    # print('Debug:')
    # debug(new, new_f)

    fitness = np.hstack((surv_f, new_f)) 
    pop = np.vstack((surv, new))  
    # print('pop:', '\n',  pop)

plt.plot(range(ITERS), log_avg, label='Mean')
plt.plot(range(ITERS), log_min, label='Best')
plt.legend()
plt.show()
