import problems
import permu_utils as putils
from optimizers import UMDA
import numpy as np
import matplotlib.pyplot as plt

PERMU_LENGTH = 7
POP_SIZE = PERMU_LENGTH*150
SURV_RATE = .5
ITERS = 70
TIMEOUT = 3*1000
INSTANCE_NAME = 'qap7_01'
LR = .15 # Learning rate 

umda = UMDA()
qap = problems.QAP(PERMU_LENGTH)

dist, flow = qap.load_instance(INSTANCE_NAME)

# Create population
pop = putils.random_population(PERMU_LENGTH,
                                    POP_SIZE)
n_surv = int(POP_SIZE*SURV_RATE) # Number of survivor solutions

# Initialization of the prob. distribution 
p_ = np.zeros((PERMU_LENGTH, PERMU_LENGTH)) 

# Init loggers
log_min = []
log_avg = []

# Evaluate for the initial population
fitness = np.empty(POP_SIZE)
for indx in range(POP_SIZE):
    fitness[indx] = qap.evaluate(pop[indx], dist, flow)

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

    # Learn a probability distribution from survivors
    p = umda.learn_distribution(surv)
    
    if LR != None:
        # Apply learning rate
        p = LR*p_ + (1-LR)*p
        p_ = p # Pi-1 = Pi
    
    # Sample new solutions
    try:
        new = umda.sample_population(p, n_surv, pop=np.array([]), 
                                     timeout=TIMEOUT)

    # except TimeoutError as e:
    except Exception as e:
        print(e)
        # Plot results
        plt.plot(range(iter_+1), log_avg, label='Mean')
        plt.plot(range(iter_+1), log_min, label='Best')
        plt.title('Ad-hoc ' + INSTANCE_NAME 
                  + ' best: {:0.2f}'.format(min(fitness)))
        plt.legend()
        plt.grid(True)
        plt.show()
        quit()

    # Evaluate the sampled solutions
    new_f = np.empty(n_surv) # Fitness of  
    for i in range(n_surv):
        new_f[i] = qap.evaluate(new[i], dist, flow)

    fitness = np.hstack((surv_f, new_f)) 
    pop = np.vstack((surv, new))  

# Plot results
plt.plot(range(ITERS), log_avg, label='Mean')
plt.plot(range(ITERS), log_min, label='Best')
plt.legend()
plt.title('Ad-hoc ' + INSTANCE_NAME 
          + ' best: {:0.2f}'.format(min(fitness)))
plt.grid(True)
plt.show()
