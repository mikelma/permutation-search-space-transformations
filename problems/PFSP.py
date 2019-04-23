import numpy as np
import os

class PFSP():

    def __init__(self, instances_dir='instances/PFSP'):
        """PFSP problem initializer.

        Args:
            instances_dir (str): Default: 'instances/PFSP'. Directory where
                                 the PFSP instaces are located.
        """
        # Chance current working directory to the file where instances are
        os.chdir(instances_dir)

    def load_instance(self, instance_name):
        """Loads saved PFSP instance.
        
        Args: 
            instance_name (str): Instance file name.
       
        Returns:
            ndarray: PFSP instance matrix (machines x jobs).
        """
        f = open(instance_name, 'r')
        lines = f.readlines()
        f.close()
        
        # Find instance size
        row = []
        for item in lines[1].split(' '):
            try:
                row.append(int(item))
            except:
                pass
        
        n_jobs = row[0]
        n_machines = row[1]

        lines = lines[3:]
        
        instance = np.empty((n_machines, n_jobs), dtype=np.int)

        # Read instance matrix        
        for i in range(n_machines):
            row_ = lines[i].strip('\n').split(' ')
            row = []
            # Check for valid data in the row string
            for item in row_:
                try:
                    row.append(int(item))
                except:
                    pass

            for j in range(n_jobs):
                instance[i][j] = row[j]

        return instance

    # def generate_instance(self, instance_name,
    #                       n_machines, n_jobs, 
    #                       min_val, max_time):
    #     """Generates a file containing a PFSP instance.
    #     The instance matrix is randomly generated with the 
    #     specified ranges and size. Matrix: machines x jobs. 

    #     Args:
    #         instance_name (str): The name of the output file.
    #         n_machines (int): Number of machines of the instance.
    #         n_jobs (int): Number of jobs of the instance.
    #         min_val (float): Lower range of the instance matrix.
    #         max_val (float): Upper range of the instance matrix.
    #     """
    #     instance = np.random.uniform(min_val, max_val,
    #                                  size=(n_machines, n_jobs))

    # def evaluate():
    #     pass

if __name__ == '__main__':

    pfsp = PFSP()
    help(pfsp)
    
    # a = pfsp.load_instance('tai20_5_0.fsp')
    # print(a)

