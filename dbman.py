import glob
import configparser
import csv
import numpy as np
import matplotlib.pyplot as plt
import datetime
import uuid

import problems
from optimizers import UMDA

from algorithm import Algorithm

import pandas as pd

class DBMan():
    
    def __init__(self, 
                 config_f='config.cfg'):
        '''DBMan constructor. 

        Args:
            config_f (str): Configuration file.

        Returns:
            DBMan instance.
        '''
        self.config_f = config_f
        self.config = configparser.ConfigParser()

        self.main_log_fields = ['id', 'date', 'problem name', 'instance','max iterations',
                                'iterations', 'space', 'sampling', 'pop size', 'check repeat']

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
            'sampling': 'ad-hoc-laplace',
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

        self.config['DATA'] = {
            'save log': 'True',
            'plot': 'False',
            'db path':'db/QAP/'}

        with open(config_f, 'w') as configfile:
            self.config.write(configfile) 
            configfile.close()

    def _read_config(self):
        '''Reads the configuration file given as a constructor parameter.

        Returns:
            Configuration file instance.
        '''
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

        umda = UMDA()
        
        config = self._read_config()

        print('[*] Config file read succsessfully.')
        
        repetitions = int(config['MAIN']['repetitions'])
        space = config['MAIN']['search space']
        sampling = config['MAIN']['sampling']
        pop_size = int(config['MAIN']['population size'])
        surv_rate = float(config['MAIN']['survivor rate'])
        iterations = int(config['MAIN']['iterations'])
        check_repeat = config['MAIN']['check repeat'] == 'True'
        timeout = int(config['MAIN']['timeout'])
        permu_dtype = config['MAIN']['permutation dtype']

        max_iterations = iterations

        problem_name = config['INSTANCE']['problem']
        instance_path = config['INSTANCE']['path']
        size = int(config['INSTANCE']['size'])

        db_path = config['DATA']['db path']
        save_log = config['DATA']['save log'] == 'True'
        plot = config['DATA']['plot'] == 'True'
        
        # Dtype
        if permu_dtype == 'int8':
            permu_dtype = np.int8

        elif permu_dtype == 'int16':
            permu_dtype = np.int16

        elif permu_dtype == 'int32':
            permu_dtype = np.int32

        # Problem
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

        # Sampling function
        if sampling == 'ad-hoc-laplace':
            sampling_func = umda.sample_ad_hoc_laplace

        elif sampling == 'no-restriction':
            sampling_func = umda.sample_no_restriction

        else:
            print('Error! ', sampling_func, ' was not found.')
            quit()

        # Repetitions loop 
        for repetition in range(repetitions):

            algorithm_id = str(uuid.uuid4())

            alg = Algorithm(size=size,
                            pop_size=pop_size,
                            evaluator=evaluator,
                            surv_rate=surv_rate,
                            iters=iterations,
                            space=space,
                            sampling_func=sampling_func,
                            timeout=timeout,
                            check_repeat=check_repeat,
                            permu_dtype=permu_dtype)

            log = alg.run()
            
            if save_log:

                data = pd.DataFrame.from_dict(log)
                data.to_csv(db_path+algorithm_id)

                iters = len(log['min'])

                main_log = {
                    'id':algorithm_id,
                    'date': str(datetime.datetime.now()),
                    'problem name': problem_name,
                    'instance': instance_path,
                    'max iterations':iterations,
                    'iterations': iters,
                    'space':space,
                    'sampling':sampling,
                    'pop size': pop_size,
                    'check repeat': check_repeat}
                
                try:
                    csvfile = open(db_path+'main.csv', 'a')
                    
                except:
                    print('The main log was not found in '+db_path+' creating a new one.')
                    self.generate_main_log()

                    csvfile = open(db_path+'main.csv', 'a')
                
                writer = csv.DictWriter(csvfile, 
                                        fieldnames=self.main_log_fields)
                writer.writerow(main_log)

                csvfile.close()

            if plot:
                plt.plot(range(iters), log['min'], label='min')
                plt.plot(range(iters), log['max'], label='max')
                plt.plot(range(iters), log['median'], label='median')
                plt.legend()
                plt.show()

    def generate_main_log(self):
        ok = False
        while not ok:
            path = input('Please enter the path for the main logger >')
            path += 'main.csv'
            if input('Are you sure? [N/y] ') == 'y':
                ok = True

        with open(path, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, 
                                    fieldnames=self.main_log_fields)
            writer.writeheader()


    def plot_main(self):

        import seaborn as sns

        path = input('Please enter the path for the main logger >')
        
        main = pd.read_csv(path+'main.csv')

        ids = list(main['id'])

        instance = list(main['instance'])[0]
        # print(ids)

        frames = []
        for id_ in ids:
            data = pd.read_csv(path+id_)

            space = str(list(main.loc[main['id']==id_]['space'])[0])
            iters = list(range(len(data)))

            data = data.assign(space = space) 
            data = data.assign(iteration = iters) 

            frames.append(data)
        
        results = pd.concat(frames)

        sns.set(style="darkgrid")

        sns.lineplot(x='iteration', y='min',
             hue="space", 
             data=results)

        plt.title(instance)
        plt.show()

    def plot_experiment(self, path):

        data = pd.read_csv(path)
        iters = list(range(len(data)))

        plt.plot(iters, list(data['max']), label='max')
        plt.plot(iters, list(data['min']), label='min')
        plt.plot(iters, list(data['median']), label='median')
        plt.legend()
        plt.title('Best result: '+str(min(data['min'])))

        plt.show()

if __name__ == '__main__':

    dbman = DBMan()

    print('[1] Generate config.')
    print('[2] Create main logger.')
    print('[3] Run experiment from config.')
    print('[4] Plot main results.')
    print('[5] Plot experiment result.')

    print('\n[0] Exit.') 

    sel = int(input('Select an option >'))
    print('')
    
    if sel == 1:
        dbman.create_config()

    elif sel == 2:
        dbman.generate_main_log()

    elif sel == 3:
        if input('[*] Run experiment [Y/n]') != 'n':
            dbman.run_experiment()

    elif sel == 4:
        dbman.plot_main()

    elif sel == 5:
        path = input('[*] Path and name of the experiment to plot >')
        dbman.plot_experiment(path)

    elif sel == 0:
        quit()
