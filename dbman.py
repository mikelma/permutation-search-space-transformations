import seaborn as sns
import glob
import configparser
import csv
import numpy as np
import matplotlib.pyplot as plt

import problems

from algorithm import Algorithm

class DBMan():
    
    def __init__(self, 
                 config_f='config.cfg'):

        self.config_f = config_f
        self.config = configparser.ConfigParser()

    def create_config(self, config_f=None):
        '''Creates a defaut configuration file. 
        If the path is not specified, it will be created as: config.cfg.

        Args:
            config_f (str or None) : Path where the config file will be created. Default: None.
        '''
        if type(config_f) is not str:
            config_f = self.config_f

        self.config['MAIN'] = {
            'repetitions': '1',
            'search space': 'permutation',
            'population size': '200',
            'survivor rate': '0.5',
            'iterations': '400',
            'check repeat':'True',
            'timeout': '3000',
            'permutation dtype': 'int8'}
        
        self.config.add_section('INSTANCE')
        self.config.set('INSTANCE', 'problem', 'QAP')
        self.config.set('INSTANCE', 'path', 'instances/QAP/tai20b.dat')
        self.config.set('INSTANCE', 'size', '20')

        with open(config_f, 'w') as configfile:
            self.config.write(configfile) 
            configfile.close()

    def _read_config(self):
        # Read config file
        try:
            self.config.read(self.config_f)
            p = self.config['MAIN']['search space']

        except Exception as e:
            print('[!] Error: The configuration file was not found in: ', 
                   self.config_f)
            print(e)
            quit()

        return self.config

    def run_experiment(self):
        
        config = self._read_config()

        print('[*] Config file read succsessfully.')
        
        repetitions = int(config['MAIN']['repetitions'])
        space = config['MAIN']['search space']
        pop_size = int(config['MAIN']['population size'])
        surv_rate = float(config['MAIN']['survivor rate'])
        iterations = int(config['MAIN']['iterations'])
        check_repeat = bool(config['MAIN']['check repeat'])
        timeout = int(config['MAIN']['timeout'])
        permu_dtype = config['MAIN']['permutation dtype']

        problem_name = config['INSTANCE']['problem']
        instance_path = config['INSTANCE']['path']
        size = int(config['INSTANCE']['size'])

        if permu_dtype == 'int8':
            permu_dtype = np.int8

        # import os
        # print(os.getcwd())
        # print(instance_path)

        if  problem_name == 'QAP':
            problem = problems.QAP() # Init problem
            dist, flow = problem.load_instance(instance_path) # Read instance

            def evaluator(permu):
                return problem.evaluate(permu, dist, flow) 


        elif problem_name == 'PFSP':
            problem = problems.PFSP() # Init problem
            instance = problem.load_instance(instance_path) # Read instance

            def evaluator(permu):
                # NOTE: Set makespan True to optimize PFSP makespan, else TFT will be evaluated
                return problem.evaluate(permu, instance, makespan=False) 

        else:
            print('Problem ', problem, ' found in ', 
                  self.config_f, ' is not a valid problem name')

        alg = Algorithm(size=size,
                        pop_size=pop_size,
                        evaluator=evaluator,
                        surv_rate=surv_rate,
                        iters=iterations,
                        space=space,
                        timeout=timeout,
                        check_repeat=check_repeat,
                        permu_dtype=permu_dtype)

        log = alg.run()
        
        iters = range(len(log['min']))

        plt.plot(iters, log['min'], label='min')
        plt.plot(iters, log['max'], label='max')
        plt.plot(iters, log['median'], label='median')
        plt.legend()

        plt.show()

if __name__ == '__main__':

    dbman = DBMan()

    if input('[*] Create config? [y/N] ') == 'y':
        dbman.create_config()

    dbman.run_experiment()
