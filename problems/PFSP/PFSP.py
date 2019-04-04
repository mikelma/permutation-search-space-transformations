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
        """Loads saved PFSP instance (.npz file).
        
        Args: 
            instance_file (str): Instance file name. .npz extension is 
                                 added if is missing.
       
        Returns:
            numpy matrix: Time-table of the loaded instance.
        """
        if '.npz' not in instance_file:
            instance_file += '.npz'

        instance = np.load(instance_file)
        return instance['arr_0']

    def generate_instance(self, instance_name,
                          min_time, max_time):
        pass

    def evaluate():
        pass

if __name__ == '__main__':

    pfsp = PFSP(5)
    help(pfsp)
