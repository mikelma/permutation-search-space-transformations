import numpy as np
import os

class QAP():

    def __init__(self, size,
                 instances_dir='instances/QAP'):
        """ QAP problem initializer.

        Args:
            size (int): Size of the problem, number of jobs
            instances_dir (str): Default: 'instances/QAP'. Directory where QAP 
                                 instaces are located
        """
        self.size = size

        # Chance current working directory to the file where instances are
        os.chdir(instances_dir)


    def load_instance(self, instance_file):
        """Loads saved QAP instance (.npz file).
        
        Args: 
            instance_file (str): Instance file name. .npz extension is 
                                 added if is missing.
       
        Returns:
            tuple: (distance_matrix, flow_matrix).
        """
        if '.npz' not in instance_file:
            instance_file += '.npz'

        instance = np.load(instance_file)
        return instance['arr_0'], instance['arr_1']
        

    def generate_instance(self, instance_name,
                          min_distance, max_distance,
                          min_flow, max_flow):
        """Generates a .txt file containing a QAP instance.
        The distance and flow matrix are randomly generated
        with the specified ranges.

        Args:
            instance_name (str): The name of the output file.
            min_distance (float): Lower range of the distance matrix.
            max_distance (float): Upper range of the distance matrix.
            min_flow (float): Lower range of the flow matrix.
            max_flow (float): Upper range of the flow matrix.
        """
        distance_matrix = np.random.uniform(min_distance, max_distance,
                                            size=(self.size, self.size))
        flow_matrix = np.random.uniform(min_flow, max_flow,
                                        size=(self.size, self.size))

        np.savez(instance_name, distance_matrix, flow_matrix)

    def evaluate(self, perm, distance_matrix, flow_matrix):
        """Evaluates the given permutation for the QAP problem.

        Args:
            perm: Permutation to evaluate.
            distance_matrix: Matrix of distances between cities.
            flow_matrix: The flow matrix.

        Returns: 
            float: fitness value of the given permutation.
        """ 
        fitness = 0

        for i in range(self.size):
            for j in range(self.size):

                factA = perm[i]
                factB = perm[j]

                distAB = distance_matrix[i][j]
                flowAB = flow_matrix[factA][factB]

                fitness += distAB*flowAB			

        return fitness

if __name__ == '__main__':
    
    qap = QAP(5)

    qap.generate_instance('qap5',
                          1, 10,
                          0, 1)
    #instance = qap.load_instance('qap5')
    #print(instance)

    distance_matrix, flow_matrix = qap.load_instance('qap5')
    print('distances:', '\n', distance_matrix, '\n')
    print('flow matrix:','\n', flow_matrix, '\n')

    fitness = qap.evaluate([0,1,2,3,4],  distance_matrix, flow_matrix)
    print('Test fitness: ', fitness)

    fitness = qap.evaluate([1,2,3,0,4],  distance_matrix, flow_matrix)
    print('Test fitness: ', fitness)


