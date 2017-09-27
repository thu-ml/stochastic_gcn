import os, sys

#datasets    = ['cora', 'citeseer', 'pubmed']
datasets    = ['ppi']
preprocess  = ['True', 'False']
det_dropout = ['True', 'False']
deg_alpha   = [(20, 1), (1, 1), (1, -1)]

f = open('run.sh', 'w')
for data in datasets:
    for pp in preprocess:
        for dropout in det_dropout:
            for deg, a in deg_alpha:
                log_file = 'logs/{}_pp{}_dropout{}_deg{}_a{}.log'.format(data, pp, dropout, deg, a)
                dropout_str = ''
                if det_dropout:
                    dropout_str += '--num_reps 4 --det_dropout'

#                command = \
#'stdbuf -o 0 python ../gcn/train.py \
#--early_stopping=1000000 --data=300000 --epochs=400 \
#--dataset={} --preprocess={} --degree={} --alpha={} {} | tee {}'.format(
#        data, pp, deg, a, dropout_str, log_file)
                command = \
'stdbuf -o 0 python ../gcn/train.py --normalization graphsage --weight_decay 0 --dropout 0.2 --layer_norm --batch_size 512 --hidden1 512 --num_fc_layers 2 \
--early_stopping=1000000 --data=40000000 --epochs=200 \
--dataset={} --preprocess={} --degree={} --alpha={} {} | tee {}'.format(
        data, pp, deg, a, dropout_str, log_file)
                f.write(command+'\n')
