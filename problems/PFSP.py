import numpy as np
import os

class PFSP():

    def __init__(self, instances_dir='instances/PFSP'):
        """PFSP problem initializer.
        """
        pass

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

    def evaluate(self, permu, times):

        n_machines = times.shape[0]

        b = [0]*n_machines

        dbug = ['']*n_machines

        for job_i, job in enumerate(permu):
            for machine in range(n_machines):
                
                if job_i == 0 and machine == 0:
                    pt = times[machine][job]
                
                elif job_i > 0 and machine == 0:
                    pt = b[machine] + times[machine][job]
                    # pt = times[machine][permu[job_i-1]] + times[machine][job]

                elif job_i == 0 and machine > 0:
                    pt = b[machine-1] + times[machine][job] 

                elif job_i > 0 and machine > 0:
                    # pt = max(b[machine-1], b[machine]) + times[machine][job]
                    pt = max(b[machine-1], times[machine][permu[job_i-1]]) + times[machine][job]

                b[machine] += pt

                for i in range(n_machines):
                    if i == machine:
                        dbug[i] += '*'*times[machine][job]
                    elif i > machine:
                        dbug[i] += '-'*times[machine][job]
                    #dbug[i] += ','

        for e in dbug:
            print(e)

        # return tft
        return pt

if __name__ == '__main__':

    pfsp = PFSP()

    instance = pfsp.load_instance('instances/PFSP/test.dat')
    permu = list(range(5)) 
    fitness = pfsp.evaluate(permu,
                        instance)

    print('* FITNESS: ', fitness)
