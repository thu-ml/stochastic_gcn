import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import os, sys
import numpy as np

data_dir   = 'logs'
datasets   = ['cora', 'citeseer', 'pubmed', 'ppi']

exps = [(20, 1, True, True, 'k', 'B=20, preprocess, dropout'),
        (20, 1, False, True, 'k:', 'B=20, preprocess, DA'),
        (1, 1, True, True, 'r', 'B=1, preprocess, dropout'),
        (1, 1, True, False, 'r:', 'B=1, nopreprocess, dropout'),
        (1, -1, False, True, 'b', 'VR, B=1, preprocess, DA'),
        (1, -1, True, False, 'b:', 'VR, B=1, nopreprocess, DA')]

for data in datasets:
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    cnt = 0
    handles = []
    for deg, a, dropout, pp, style, _ in exps:
        log_file = 'logs/{}_pp{}_dropout{}_deg{}_a{}.log'.format(
                    data, pp, dropout, deg, a)
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

        l = ax[0,0].plot(losses, style, alpha=0.5)
        ax[0,1].plot(acc,    style, alpha=0.5)
        ax[1,0].plot(amt_data, losses, style, alpha=0.5)
        ax[1,1].plot(amt_data, acc,    style, alpha=0.5)
        handles.append(l[0])
        cnt += 1

    ax[0,2].legend(handles, [l[-1] for l in exps])
    ax[0,2].axis('off')
    ax[1,2].axis('off')

    ax[0,0].set_xlabel('Number of iterations')
    ax[0,1].set_xlabel('Number of iterations')
    ax[1,0].set_xlabel('Amount of data seen')
    ax[1,1].set_xlabel('Amount of data seen')
    ax[0,0].set_ylabel('Validation loss')
    ax[0,1].set_ylabel('Validation accuracy')
    ax[1,0].set_ylabel('Validation loss')
    ax[1,1].set_ylabel('Validation accuracy')
    ax[0,0].set_xlim([0, 200])
    ax[0,1].set_xlim([0, 200])
    ax[1,0].set_xlim([0, 250000])
    ax[1,1].set_xlim([0, 250000])
    ylim = (0, 0)
    if data=='cora':
        ylim = (0.7, 0.81)
    elif data=='pubmed':
        ylim = (0.7, 0.81)
    elif data=='citeseer':
        ylim = (0.65, 0.75)
    if ylim[0] != 0:
        ax[0,1].set_ylim(ylim)
        ax[1,1].set_ylim(ylim)
    
    fig.savefig('{}.pdf'.format(data))
    
