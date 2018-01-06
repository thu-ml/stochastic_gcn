import os, sys
import numpy as np

exps = [#(20, 'False', 'True', False, 'Exact'),
        (20, 'False', 'True', False, 'Exact'),
        (1,  'False', 'True', False, 'NS'),
        (1,  'False', 'True', True,  'NS+PP'),
        (1,  'True',  'True', True,  'NS+PP+CV'),
        (1,  'TrueD', 'True', True,  'NS+PP+CVD')]   

# Epochs, time, amt. of data, sparse gflop, dense gflop
dir      = 'logs'
dataset  = 'reddit3'
accuracy = 0.94
num_runs = 1#5

for deg, cv, dropout, pp, text in exps:
    accs   = []
    epochs = []
    times  = []
    data   = []
    sflops = []
    dflops = []
    for run in range(num_runs):
        log_file = '{}/{}_pp{}_dropout{}_deg{}_cv{}_run{}.log'.format(
                    dir, dataset, pp, dropout, deg, cv, run)
        e  = 0
        t  = 0
        d  = 0
        sf = 0
        df = 0
        best_acc = 0
        all_times = []

        with open(log_file) as f:
            lines = f.readlines()
            will_end = False
            for line in lines:
                if line.find('Epoch') != -1:
                    line = line.replace('=', ' ').split()
                    tt   = float(line[17]) - float(line[19])
                    all_times.append(tt)
                    t    += tt
                    acc  = float(line[12])
                    d    = float(line[-1])
                    #print(d)
                    if acc >= accuracy:
                        will_end = True
                    best_acc = max(best_acc, acc)
                    e += 1
                elif line.find('FLOPS') != -1:
                    line = line.replace(',', '').split()
                    sf   += float(line[11])
                    df   += float(line[15])
                    if will_end:
                        break

        if not will_end:
            accs.append(best_acc)
        else:
            accs.append(accuracy)

        epochs.append(e)
        times.append(t)
        data.append(d)
        sflops.append(sf)
        dflops.append(df)

    print('{} & {:.3f} & {} & {:.0f} & {:.3g} & {:.3g}\\\\'.format(
          text, np.mean(accs), np.mean(epochs), np.mean(times), #np.mean(data)/1000000, 
          np.mean(sflops), np.mean(dflops)/1024))
    #print(np.mean(all_times), np.std(all_times))
