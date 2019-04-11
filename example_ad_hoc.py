import problems
import optimizers
import permu_utils as putils
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

def plot(m):
    plt.matshow(p)
    # Loop over data dimensions and create text annotations.
    for i in range(len(m)):
        for j in range(len(m)):
            plt.text(j, i, np.around(p[i, j], 1),
                     ha="center", va="center", color="w")
    plt.draw()
    plt.pause(.5)
    plt.close()

# PERMU_LENGTH = 9
PERMU_LENGTH = 5
# POP_SIZE = PERMU_LENGTH*100
POP_SIZE = PERMU_LENGTH*10
SURV_RATE = .5
ITERS = 150
TIMEOUT = 10*1000
# INSTANCE_NAME = 'qap9_01'
INSTANCE_NAME = 'qap5_01'
SHOW_PLOT = False
LR = .15 # Learning rate 

umda = optimizers.UMDA()
qap = problems.QAP(PERMU_LENGTH)

dist, flow = qap.load_instance(INSTANCE_NAME)

# Create population
pop = putils.random_population(PERMU_LENGTH,
                                    POP_SIZE)
n_surv = int(POP_SIZE*SURV_RATE) # Number of survivor solutions

# Initialization of the prob. distribution 
p_ = np.zeros((PERMU_LENGTH, PERMU_LENGTH)) 

# Evaluate for the initial population
fitness = np.empty(POP_SIZE)
for indx in range(POP_SIZE):
    fitness[indx] = qap.evaluate(pop[indx], dist, flow)

for iter_ in range(ITERS):

    print('mean fitness: ', np.mean(fitness), ' best: ', min(fitness))
    
    # Select best solutions
    surv = np.empty((n_surv, PERMU_LENGTH), dtype=np.int)
    surv_f = np.empty(n_surv) # Fitness of survivors 
    for i in range(n_surv):
        bests_indx = np.argmin(fitness)
        surv[i] = pop[bests_indx]
        surv_f[i] = fitness[bests_indx]

        pop = np.delete(pop, bests_indx, axis=0)
        fitness = np.delete(fitness, bests_indx)

    # Learn a probability distribution from survivors
    p = umda.learn_distribution(surv)
    
    if LR != None:
        # Apply learning rate
        p = LR*p_ + (1-LR)*p
        p_ = p # Pi-1 = Pi
    
    if SHOW_PLOT:
        plot(p)

    # Sample new solutions
    new = umda.sample_population(p, n_surv, pop=np.array([]), 
                                 timeout=TIMEOUT, quit_in_timeout=True)

    # Evaluate the sampled solutions
    new_f = np.empty(n_surv) # Fitness of  
    for i in range(n_surv):
        new_f[i] = qap.evaluate(new[i], dist, flow)

    fitness = np.hstack((surv_f, new_f)) 
    pop = np.vstack((surv, new))  

