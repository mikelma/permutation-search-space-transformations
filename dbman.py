import seaborn as sns
import glob
import configparser

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
            'search space': 'permutation',
            'population size': '200',
            'survivor rate': '0.5',
            'iterations': '400',
            'timeout': '3'}
        
        self.config.add_section('INSTANCE')
        self.config.set('INSTANCE', 'problem', 'QAP')
        self.config.set('INSTANCE', 'path', 'instances/QAP/tai20_5.dat')
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

    def run(self):
        
        config = self._read_config()

        print('[*] Config file read succsessfully.')

        space = config['MAIN']['search space']
        pop_size = int(config['MAIN']['population size'])
        surv_rate = float(config['MAIN']['survivor rate'])
        iterations = int(config['MAIN']['iterations'])
        timeout = int(config['MAIN']['timeout'])

        problem = config['INSTANCE']['problem']
        instance_path = config['INSTANCE']['path']
        size = int(config['INSTANCE']['size'])

        assert space in ['vj', 'permutation'], 'Specified search space in '+self.config_f+' is not valid.'
        
        if  problem == 'QAP':
            pass

        elif problem == 'PFSP':
            pass
        else:
            print('Problem ', problem, ' found in ', 
                  self.config_f, ' is not a valid problem name')
if __name__ == '__main__':

    dbman = DBMan()

    if input('[*] Create config? [y/N] ') == 'y':
        dbman.create_config()

    dbman.run()
