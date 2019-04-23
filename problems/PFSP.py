import numpy as np
import os

class PFSP():

    def __init__(self, size, 
                 instances_dir='instances/PFSP'):
        """ PFSP problem initializer.

        Args:
            size (int): Size of the problem, number of jobs
            instances_dir (str): Default: 'instances/PFSP'. Directory where
                                 the PFSP instaces are located.
        """
        self.size = size

        # Chance current working directory to the file where instances are
        os.chdir(instances_dir)

    def load_instance(self, instance_file):
        """Loads saved PFSP instance.
        
        Args: 
            instance_name (str): Instance file name.
       
        Returns:
            ndarray: PFSP instance matrix (machines x jobs).
        """
        f = open(instance_name, 'r')
        lines = f.readlines()
        f.close()
        sizes = lines[0].strip('\n').split(',') 
        n_machines = int(sizes[0]) 
        n_jobs = int(sizes[1]) 

        # Distance matrix 
        d = lines[1].split(' ')  
        instance = np.empty((n_machines, n_jobs))
        a = 0
        for i in range(n_machines):
            for j in range(n_jobs):
                instances[i][j] = float(d[a])
                a += 1
        return instance

    def generate_instance(self, instance_name,
                          n_machines, n_jobs, 
                          min_val, max_time):
        """Generates a file containing a PFSP instance.
        The instance matrix is randomly generated with the 
        specified ranges and size. Matrix: machines x jobs. 

        Args:
            instance_name (str): The name of the output file.
            n_machines (int): Number of machines of the instance.
            n_jobs (int): Number of jobs of the instance.
            min_val (float): Lower range of the instance matrix.
            max_val (float): Upper range of the instance matrix.
        """
        instance = np.random.uniform(min_val, max_val,
                                     size=(n_machines, n_jobs))

    def evaluate():
        pass

if __name__ == '__main__':

    pfsp = PFSP(5)
    help(pfsp)
