from __future__ import division
from __future__ import print_function

from time import time
import sys
import tensorflow as tf

from gcn.utils import *
from gcn.plaingcn import PlainGCN
from gcn.vrgcn import VRGCN
from scheduler import PyScheduler
from tensorflow.contrib.opt import ScipyOptimizerInterface
import scipy.sparse as sp
from stats import Stat

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('normalization', 'gcn', 'gcn or graphsage')
flags.DEFINE_integer('max_degree', -1, 'Subsample the input. Maximum number of degree. For GraphSAGE.')

# Load data
num_data, train_adj, full_adj, features, train_features, test_features, labels, train_d, val_d, test_d = \
        load_data(FLAGS.dataset)

full_adj = full_adj + sp.eye(full_adj.shape[0])
E1 = full_adj.nnz
E2 = full_adj.dot(full_adj).nnz
print('{} nodes, {} edges, {} features, Deg = {}, Deg2 = {}'.format(num_data, E1, features.shape[1], E1/num_data, E2/num_data))
print('{} & {} & {} & {:.1f} & {:.1f} \\\\'.format(num_data, E1, features.shape[1], E1/num_data, E2/num_data))
