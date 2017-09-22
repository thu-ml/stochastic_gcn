import os, sys

# datasets   = ('cora', 'citeseer', 'pubmed')
# datasets   = ['cora']
datasets   = ['reddit']
# preprocess = ('True', 'False')
preprocess = ['True']
# degrees    = (1, 2, 4, 10000)
degrees    = (1, 20)
alpha      = (1.0, 0.5, 0.25, 0.125)

f = open('run.sh', 'w')
for data in datasets:
    for pp in preprocess:
        for deg in degrees:
            for a in alpha:
                log_file = 'logs/{}_pp{}_deg{}_a{}.log'.format(data, pp, deg, a)
#                command = \
#'stdbuf -o 0 python ../gcn/train.py --dropout 0 \
#--early_stopping=1000000 --data=300000 --epochs=400 \
#--dataset={} --preprocess={} --degree={} --alpha={} | tee {}'.format(
#        data, pp, deg, a, log_file)
                command = \
'stdbuf -o 0 python ../gcn/train.py --dropout 0 --weight_decay 0 --hidden1 128 --normalization graphsage --learning_rate 3e-4 \
--early_stopping=10000000 --epochs=20 --data=10000000 \
--dataset={} --preprocess={} --degree={} --alpha={} | tee {}'.format(
        data, pp, deg, a, log_file)
                f.write(command+'\n')
