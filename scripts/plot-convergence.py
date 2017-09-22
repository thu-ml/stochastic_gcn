import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import os, sys
import numpy as np

data_dir   = 'logs'
# datasets   = ('cora', 'citeseer', 'pubmed')
datasets   = ['reddit']
# preprocess = ['True', 'False']
preprocess = ['True']

# exps = [(10000, 1.0, 'k', 'Batch'),
#          (1,     1.0, 'c', 'D=1'),
#          (1,     0.5, 'm', 'a=0.5, D=1')]
exps = [(1, 1.0, 'k', 'a=1'),
        (1, 0.5, 'c', 'a=0.5'),
        (1, 0.25,  'm', 'a=0.25'),
        (1, 0.125,   'y', 'a=0.125')]

for data in datasets:
    for pp in preprocess:
        fig, ax = plt.subplots(2, 2)
        cnt = 0
        handles = []
        for deg, a, style, _ in exps:
            log_file = 'logs/{}_pp{}_deg{}_a{}.log'.format(
                        data, pp, deg, a)
            losses   = []
            amt_data = []
            acc      = []
            with open(log_file) as f:
                lines = f.readlines()
                for line in lines:
                    if line.find('Epoch') != -1:
                        line = line.split()
                        losses.append(float(line[7]))
                        acc.append(float(line[9]))
                        amt_data.append(float(line[-1]))

            l = ax[0,0].plot(losses, style)
            ax[0,1].plot(acc,    style)
            ax[1,0].plot(amt_data, losses, style)
            ax[1,1].plot(amt_data, acc,    style)
            handles.append(l[0])
            cnt += 1

        fig.legend(handles, [l[3] for l in exps])

        ax[0,0].set_xlabel('Number of iterations')
        ax[0,1].set_xlabel('Number of iterations')
        ax[1,0].set_xlabel('Amount of data seen')
        ax[1,1].set_xlabel('Amount of data seen')
        ax[0,0].set_ylabel('Validation loss')
        ax[0,1].set_ylabel('Validation accuracy')
        ax[1,0].set_ylabel('Validation loss')
        ax[1,1].set_ylabel('Validation accuracy')
        #ax[0,0].set_xlim([0, 400])
        #ax[0,1].set_xlim([0, 100])
        #ax[1,0].set_xlim([0, 300000])
        #ax[1,1].set_xlim([0, 100000])
        #ax[0,1].set_ylim([0.7, 0.8])
        #ax[1,1].set_ylim([0.7, 0.8])
        ax[0,0].set_xlim([0, 20])
        ax[0,1].set_xlim([0, 20])
        ax[1,0].set_xlim([0, 1e7])
        ax[1,1].set_xlim([0, 1e7])
        ax[0,0].set_ylim([0.1, 0.3])
        ax[1,0].set_ylim([0.1, 0.3])
        ax[0,1].set_ylim([0.94, 0.96])
        ax[1,1].set_ylim([0.94, 0.96])
        
        fig.savefig('{}_{}.pdf'.format(data, pp))
        
