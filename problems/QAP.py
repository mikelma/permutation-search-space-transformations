import numpy as np
import os

class InstanceSizeError(Exception):
    def __init__(self, message):
        super().__init__(message)

class QAP():

    def __init__(self):
        """ QAP problem initializer.
        """
        pass

    def load_instance(self, instance_name):
        """Loads saved QAP instance.
        
        Args: 
            instance_name (str): Instance file name.
       
        Returns:
            tuple: (distance_matrix, flow_matrix).

        Raises:
            InstanceSizeError: Error while formatting the string from the instance 
                to a numpy array. The array has not the desired size.
        """
        f = open(instance_name, 'r')
        lines = f.readlines()
        f.close()

        size = int(lines[0].strip('\n').strip(' ')) 
        del lines[0]

        def _format(str_, size): 

            matrix = np.empty((size, size), dtype=np.int)
            for i in range(size):

                str_ = lines[i].split(' ')

                # Clean row, str -> int
                row = []
                for item in str_:
                    try:
                        row.append(int(item.strip('\n')))
                    except:
                        pass

                # Check for size errors
                if len(row) != size:
                    raise InstanceSizeError(
                        'The instance matrix created from '
                        + instance_name 
                        + ', has not the correct size '
                        + str(size) + ', instead its size is '
                        + str(len(row)) + '.')

                # Add data to the matrix 
                for j in range(size):
                    matrix[i][j] = int(row[j])

            return matrix

        distances = _format(lines, size)

        lines = lines[size:]
        
        flow = _format(lines, size)

        return distances, flow
        
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
        size = len(perm)

        for i in range(size):
            for j in range(size):

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


