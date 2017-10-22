import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import os, sys
import numpy as np
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")

datasets   = ['cora', 'pubmed', 'nell', 'citeseer', 'ppi', 'reddit']
exps = [('VarNS', 'NS'), ('VarNSPP', 'NS+PP'), ('VarCV', 'NS+PP+CV'), 
        ('DVarNS', 'NS'), ('DVarNSPP', 'NS+PP'), ('DVarCV', 'NS+PP+CV'), ('DVarCVD', 'NS+PP+CVD')]
dir = 'logs'

has_dropout = []
algos       = []
biass       = [] 
stdevs      = []
data        = []

for d in datasets:
    for e, etitle in exps:
        if etitle == 'NS' and d == 'nell':
            continue
        log_file = '{}/{}_{}.log'.format(dir, d, e)
        with open(log_file) as f:
            lines = f.readlines()
            lines = [line.replace('=', ' ').split() for line in lines if line.find('grad')!=-1]
            full_stdev = float(lines[0][-1])
            part_bias  = float(lines[1][-1])
            part_stdev = float(lines[2][-1])

            hd = e[0][0] == 'D'

            has_dropout.append(hd)
            algos.append(etitle)
            biass.append(part_bias)
            stdevs.append(part_stdev)
            data.append(d)
            if e == 'VarCV' or e == 'DVarCVD':
                has_dropout.append(hd)
                algos.append('Exact')
                biass.append(0)
                stdevs.append(full_stdev)
                data.append(d)


df = pd.DataFrame(data={'has_dropout': has_dropout, 'Algorithm': algos, 'Dataset': data, 'Gradient Bias': biass, 'Gradient Std. Dev.': stdevs})
print(df)

ddropout = df[df['has_dropout'] == True]
ndropout = df[df['has_dropout'] == False]

g = sns.factorplot(x='Dataset', y='Gradient Bias', hue='Algorithm', data=ndropout, kind='bar', aspect=2, size=2)
g.savefig('var-nb.pdf')
g = sns.factorplot(x='Dataset', y='Gradient Std. Dev.', hue='Algorithm', data=ndropout, kind='bar', aspect=2, size=2)
g.savefig('var-ns.pdf')
g = sns.factorplot(x='Dataset', y='Gradient Bias', hue='Algorithm', data=ddropout, kind='bar', aspect=2, size=2)
g.savefig('var-db.pdf')
g = sns.factorplot(x='Dataset', y='Gradient Std. Dev.', hue='Algorithm', data=ddropout, kind='bar', aspect=2, size=2)
g.savefig('var-ds.pdf')
