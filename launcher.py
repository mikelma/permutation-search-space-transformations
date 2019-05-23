import sys
import csv
import argparse
import numpy as np
import datetime

# import pandas as pd

from algorithm import Algorithm
from optimizers import UMDA
import problems


umda = UMDA()

# Parse arguments
parser = argparse.ArgumentParser(description='UMDA Launcher')

parser.add_argument('-id', help='Identifier', type=int)
parser.add_argument('-i', '--iters', help='Iterations', type=int)
parser.add_argument('-Pn', '--problem', help='Name of the problem', type=str)
parser.add_argument('-Pp', '--instance', help='Path of the instance to solve', type=str)
parser.add_argument('-Ps', '--pop-size', help='Population size', type=int, default=200)
parser.add_argument('-Sr', '--srate', help='Survivor rate', type=float, default=.5)
parser.add_argument('-s', '--space', help='Search space to work with', type=str)
parser.add_argument('-Sf', '--sampling-func', help='Sampling function to use', type=str)
parser.add_argument('-c', '--check-repeat', help='Enable check repeat', type=bool, default=True)
parser.add_argument('-d', '--dtype', help='Permutation dtype', type=str, default='int8')
parser.add_argument('-t', '--timeout', help='Timeout sampling', type=int, default=5000)
parser.add_argument('-o', '--out', help='Output file path', type=str)
parser.add_argument('-m', '--main-out', help='Main logger file path, including file name', type=str)
parser.add_argument('-v', '--verbose', help='If enabled, basic info of each iter is printed', 
                    type=str, default=False)

args = parser.parse_args()

# Define problem
if args.problem == 'QAP':
    problem = problems.QAP() # Init problem
    dist, flow = problem.load_instance(args.instance) # Read instance

    size = dist.shape[0]

    def evaluator(permu):
        return problem.evaluate(permu, dist, flow) 

elif args.problem == 'PFSP':
    problem = problems.PFSP() # Init problem
    instance = problem.load_instance(args.instance) # Read instance

    size = instance.shape[1]

    def evaluator(permu):
        # NOTE: Set makespan True to optimize PFSP makespan, else TFT will be evaluated
        return problem.evaluate(permu, instance, makespan=False) 

# Define permutation dtype
if args.dtype == 'int8':
    dtype = np.int8

elif args.dtype == 'int16':
    dtype = np.int16

elif args.dtype == 'int32':
    dtype = np.int32

# Sampling function
if args.sampling_func == 'ad-hoc-laplace':
    sampling_func = umda.sample_ad_hoc_laplace

elif args.sampling_func == 'no-restriction':
    sampling_func = umda.sample_no_restriction

else:
    print('Error! ', args.sampling_func, ' sampling function was not found.')
    quit()

# Init algorithm
alg = Algorithm(size=size,
                pop_size=args.pop_size,
                evaluator=evaluator,
                surv_rate=args.srate,
                iters=args.iters,
                space=args.space,
                sampling_func=sampling_func,
                timeout=args.timeout,
                check_repeat=args.check_repeat,
                permu_dtype=dtype)

log = alg.run(args.verbose)

# Write experiment data to logger
with open(args.out+str(args.id)+'.csv', 'w') as f:  # Just use 'w' mode in 3.x

    w = csv.DictWriter(f, log.keys())
    w.writeheader()

    for i in range(len(log['min'])):
        w.writerow({'min':log['min'][i],
                    'max':log['max'][i],
                    'mean':log['mean'][i],
                    'median':log['median'][i]})

# data = pd.DataFrame.from_dict(log)
# data.to_csv(args.out+str(args.id)+'.csv')

# Append to main logger
main_log = {
    'id':args.id,
    'date': str(datetime.datetime.now()),
    'problem name': args.problem,
    'instance': args.instance,
    'max iterations': args.iters,
    'iterations': len(log['min']),
    'space': args.space,
    'sampling': args.sampling_func,
    'pop size': args.pop_size,
    'check repeat': args.check_repeat,
    'min':log['min'][-1]}

with open(args.main_out, 'a') as f:
    w = csv.DictWriter(f, main_log.keys())
    w.writerow(main_log)
