import os, sys

datasets    = ['cora', 'citeseer', 'pubmed', 'nell', 'ppi', 'reddit']
gcn_datasets = set(['cora', 'citeseer', 'pubmed', 'nell'])
preprocess  = ['True', 'False']
dropout = [True, False]
deg_alpha   = [(20, 1), (1, 1), (1, -1)]

f = open('run.sh', 'w')
for data in datasets:
    for pp in preprocess:
        for d in dropout:
            for deg, a in deg_alpha:
                # Dropout 
                dropout_rate = 0
                if d:
                    if data in gcn_datasets:
                        dropout_rate = 0.5
                    else:
                        dropout_rate = 0.2

                log_file = 'logs/{}_pp{}_dropout{}_deg{}_a{}.log'.format(data, pp, dropout_rate, deg, a)

                if data in set(['cora', 'citeseer', 'pubmed']):
                    command = \
'stdbuf -o 0 python ../gcn/train.py \
--early_stopping=1000000 --data=50000 --epochs=400 \
--dataset={} --preprocess={} --degree={} --test_degree={} --alpha={} --dropout {} | tee {}'.format(
            data, pp, deg, deg, a, dropout_rate, log_file)
                elif data == 'nell':
                    command = \
'stdbuf -o 0 python ../gcn/train.py \
--early_stopping=1000000 --data=50000 --epochs=400 --hidden1 64 --weight_decay 1e-5 \
--dataset={} --preprocess={} --degree={} --test_degree={} --alpha={} --dropout {} | tee {}'.format(
            data, pp, deg, deg, a, dropout_rate, log_file)
                elif data == 'ppi':
                    command = \
'stdbuf -o 0 python ../gcn/train.py --normalization graphsage --weight_decay 0 --layer_norm --batch_size 512 --hidden1 512 --num_fc_layers 2 \
--early_stopping=1000000 --data=10000000 --epochs=200 \
--dataset={} --preprocess={} --degree={} --test_degree={} --alpha={} --dropout {} | tee {}'.format(
            data, pp, deg, deg, a, dropout_rate, log_file)
                else:
                    command = \
'stdbuf -o 0 python ../gcn/train.py --normalization graphsage --weight_decay 0 --layer_norm --batch_size 512 --hidden1 128 --num_fc_layers 2 \
--early_stopping=1000000 --data=40000000 --epochs=30 \
--dataset={} --preprocess={} --degree={} --test_degree={} --alpha={} --dropout {} | tee {}'.format(
            data, pp, deg, deg, a, dropout_rate, log_file)
                f.write(command+'\n')
