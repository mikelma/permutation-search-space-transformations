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


    def load_instance(self, instance_name):
        """Loads saved QAP instance.
        
        Args: 
            instance_name (str): Instance file name.
       
        Returns:
            tuple: (distance_matrix, flow_matrix).
        """
        f = open(instance_name, 'r')
        lines = f.readlines()
        f.close()
        size = int(lines[0].strip('\n')) 

        # Distance matrix 
        d = lines[1].split(' ')  
        distances = np.empty((size, size))
        a = 0
        for i in range(size):
            for j in range(size):
                distances[i][j] = float(d[a])
                a += 1

        # Flow matrix 
        f = lines[2].split(' ')  
        flow = np.empty((size, size))
        a = 0
        for i in range(size):
            for j in range(size):
                flow[i][j] = float(d[a])
                a += 1

        return distances, flow
        

    def generate_instance(self, instance_name,
                          min_distance, max_distance,
                          min_flow, max_flow):
        """Generates a file containing a QAP instance.
        The distance and flow matrix are randomly generated
        with the specified ranges. Matrix order: distance, flow.

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

        str_ = str(self.size) + '\n'
        for i in range(self.size):
            for j in range(self.size):
                str_ += str(distance_matrix[i][j]) + ' '
        str_ = str_[:-1]
        str_ += '\n'
        for i in range(self.size):
            for j in range(self.size):
                str_ += str(flow_matrix[i][j]) + ' '
        str_ = str_[:-1]

        f = open(instance_name, 'w')
        f.write(str_)
        f.close()

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
    d, f = qap.load_instance('qap5')
    print('d: ', d)
    print('f: ', f)
    quit()
    #instance = qap.load_instance('qap5')
    #print(instance)

    distance_matrix, flow_matrix = qap.load_instance('qap5')
    print('distances:', '\n', distance_matrix, '\n')
    print('flow matrix:','\n', flow_matrix, '\n')

    fitness = qap.evaluate([0,1,2,3,4],  distance_matrix, flow_matrix)
    print('Test fitness: ', fitness)

    fitness = qap.evaluate([1,2,3,0,4],  distance_matrix, flow_matrix)
    print('Test fitness: ', fitness)


