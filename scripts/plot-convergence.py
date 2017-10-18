import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import os, sys
import numpy as np
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")

#datasets   = ['cora', 'pubmed', 'nell', 'citeseer', 'ppi']
#exps1 = [(20, False, True, True,  'k',  'Batch'),
#         (1,  False, True, False, 'r:', '-CV, -PP'),
#         (1,  False, True, True,  'r',  '-CV, +PP'),
#         (1,  True,  True, False, 'b:', '+CV, -PP'),
#         (1,  True,  True, True,  'b',  '+CV, +PP')]
#exps2 = [(20, False, True,  True,  'k',  'Batch, +dropout'),
#         (20, False, False, True,  'k:', 'Batch'),
#         (1,  False, False, False, 'r:', '-CV, -PP'),
#         (1,  False, False, True,  'r',  '-CV, +PP'),
#         (1,  True,  False, False, 'b:', '+CV, -PP'),
#         (1,  True,  False, True,  'b',  '+CV, +PP')]
#all_exps = [exps1, exps2]
#num_runs = 10
#dir='logs_old'

datasets   = ['ppi']
exps1 = [(20, False, 'True', True, 'k', 'Batch'),
         (20, False, 'Fast', True, 'k', 'Batch'),
         (1,  False, 'True', False, 'r:', 'SGD'),
         (1,  False, 'True', True,  'r',  'SGD+PP'),
         (1,  False, 'Fast', True,  'b',  'SGD+PP+Det'),
         (1,  True,  'True', True,  'b',  'SGD+PP+CV'),
         (1,  True,  'Fast', True,  'g',  'SGD+PP+CV+Det')]
all_exps = [exps1]
num_runs = 3
dir='logs'

gfig, gax = plt.subplots(2, 3, figsize=(16, 8))
for ndata, data in enumerate(datasets):
    for nexp, exps in enumerate(all_exps):
        fig, ax = plt.subplots(2, 3, figsize=(16, 8))
        cnt = 0
        handles  = []
        losses   = []
        accs     = []
        train_losses = []
        types    = []
        iters    = []
        amt_data = []
        units    = []
        legends  = []
    
        cnt = 0
        for deg, cv, dropout, pp, style, text in exps:
            if data == 'nell' and not pp:
                continue
            legends.append(text)
            my_amt_data = []
            for run in range(num_runs):
                log_file = '{}/{}_pp{}_dropout{}_deg{}_cv{}_run{}.log'.format(
                            dir, data, pp, dropout, deg, cv, run)
                N = 0
                with open(log_file) as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.find('Epoch') != -1:
                            line = line.split()
                            losses.append(float(line[7]))
                            accs.append(float(line[9]))
                            train_losses.append(float(line[3]))
                            N += 1
                            if N > len(my_amt_data):
                                my_amt_data.append(float(line[-1]))
                            units.append(run)
    
                amt_data.extend(my_amt_data[:N])
                iters.extend(range(N))
                types.extend([cnt]*N)
    
            cnt += 1
    
        df = pd.DataFrame(data={'loss': losses, 'acc': accs, 'type': types, 'iter': iters, 'data': amt_data, 'run': units, 'train_loss': train_losses})
    
        # iter - trainloss
        sns.tsplot(data=df, time="iter", unit="run", condition="type", value="train_loss", ax=ax[0,0], legend=False)
        # data - trainloss
        sns.tsplot(data=df, time="data", unit="run", condition="type", value="loss", ax=ax[1,0], legend=False)
        # iter - loss
        sns.tsplot(data=df, time="iter", unit="run", condition="type", value="loss", ax=ax[0,1], legend=False)
        # data - loss
        sns.tsplot(data=df, time="data", unit="run", condition="type", value="loss", ax=ax[1,1], legend=False)
        # iter - acc
        sns.tsplot(data=df, time="iter", unit="run", condition="type", value="acc", ax=ax[0,2], legend=False)
        # data - acc
        sns.tsplot(data=df, time="data", unit="run", condition="type", value="acc", ax=ax[1,2], legend=False)

        ggax = gax[ndata//3, ndata%3]
        sns.tsplot(data=df, time="iter", unit="run", condition="type", value="acc", ax=ggax)
        ggax.set_title(data)
    
        ax[0,2].legend(ax[0,0].lines, legends)
        #ax[0,2].axis('off')
        #ax[1,2].axis('off')
    
    
        ax[0,0].set_xlabel('Number of iterations')
        ax[0,1].set_xlabel('Number of iterations')
        ax[0,2].set_xlabel('Number of iterations')
        ggax.set_xlabel('Number of iterations')
        ax[1,0].set_xlabel('Amount of data seen')
        ax[1,1].set_xlabel('Amount of data seen')
        ax[1,2].set_xlabel('Amount of data seen')
        ax[0,0].set_ylabel('Training loss')
        ax[0,1].set_ylabel('Validation loss')
        ax[0,2].set_ylabel('Validation accuracy')
        ggax.set_ylabel('Validation accuracy')
        ax[1,0].set_ylabel('Training loss')
        ax[1,1].set_ylabel('Validation loss')
        ax[1,2].set_ylabel('Validation accuracy')
        ylim = (0, 0)
        if data=='cora':
            ylim = (0.6, 0.81)
        elif data=='pubmed':
            ylim = (0.6, 0.81)
        elif data=='citeseer':
            ylim = (0.60, 0.75)
        elif data=='nell':
            ylim = (0.0, 0.7)
        elif data=='ppi':
            ylim = (0.9, 1.0)
        elif data=='reddit':
            ylim = (0.85, 1.0)
        else:
            ylim = (0.55, 0.7)
        if data=='ppi' or data=='reddit':
            xlim = (0, 1e7)
        else:
            xlim = (0, 100000)
        if data=='ppi':
            xlim0 = (0, 800)
        elif data=='reddit':
            xlim0 = (0, 50)
        else:
            xlim0 = (0, 200)
    
        print(ylim, xlim, xlim0)
        if ylim[0] != 0:
            ax[0,2].set_ylim(ylim)
            ggax.set_ylim(ylim)
            ax[1,2].set_ylim(ylim)
        ax[1,0].set_xlim(xlim)
        ax[1,1].set_xlim(xlim)
        ax[1,2].set_xlim(xlim)
        ax[0,0].set_xlim(xlim0)
        ax[0,1].set_xlim(xlim0)
        ax[0,2].set_xlim(xlim0)
        ggax.set_xlim(xlim0)
        
        fig.savefig('{}_{}.pdf'.format(data, nexp))

gfig.savefig('iter-acc.pdf') 
