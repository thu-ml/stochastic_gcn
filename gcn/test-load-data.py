from utils import load_graphsage_data

num_data, train_adj, full_adj, feats, labels, train_data, val_data, test_data = \
        load_graphsage_data('reddit/reddit')
