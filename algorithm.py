import configparser
import numpy as np
import permu_utils as putils

class Algorithm():
    
    def __init__(self,
                 size, 
                 pop_size,
                 surv_rate,
                 iters,
                 timeout,
                 instance,
                 problem,
                 optimizer,
                 permu_dtype=np.int8):
        
        self.size = size
        self.pop_size = pop_size
        
        self.problem = problem
        self.optimizer = oprimizer

    def run():

        # Sample initial random population
        pop = putils.random_population(self.size,
                                       self.pop_size)

# if __name__ == '__main__':
