import numpy as np
import os

class InstanceSizeError(Exception):
    def __init__(self, message):
        super().__init__(message)

class QAP():

    def __init__(self, instances_dir='instances/QAP'):
        """ QAP problem initializer.

        Args:
            instances_dir (str): Default: 'instances/QAP'. Directory where QAP 
                                 instaces are located
        """
        self.instances_dir = instances_dir
        self.wdir = os.getcwd()


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
        # Change the working dir to where instances are
        os.chdir(self.instances_dir)

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

        # return to the original working dir 
        os.chdir(self.wdir)

        return distances, flow
        

    # def generate_instance(self, instance_name,
    #                       min_distance, max_distance,
    #                       min_flow, max_flow):
    #     """Generates a file containing a QAP instance.
    #     The distance and flow matrix are randomly generated
    #     with the specified ranges. Matrix order: distance, flow.

    #     Args:
    #         instance_name (str): The name of the output file.
    #         min_distance (float): Lower range of the distance matrix.
    #         max_distance (float): Upper range of the distance matrix.
    #         min_flow (float): Lower range of the flow matrix.
    #         max_flow (float): Upper range of the flow matrix.
    #     """
    #     distance_matrix = np.random.uniform(min_distance, max_distance,
    #                                         size=(size, size))
    #     flow_matrix = np.random.uniform(min_flow, max_flow,
    #                                     size=(size, size))

    #     str_ = str(size) + '\n'
    #     for i in range(size):
    #         for j in range(size):
    #             str_ += str(distance_matrix[i][j]) + ' '
    #     str_ = str_[:-1]
    #     str_ += '\n'
    #     for i in range(size):
    #         for j in range(size):
    #             str_ += str(flow_matrix[i][j]) + ' '
    #     str_ = str_[:-1]

    #     f = open(instance_name, 'w')
    #     f.write(str_)
    #     f.close()

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


