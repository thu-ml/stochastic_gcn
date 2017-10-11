from __future__ import division
from __future__ import print_function

from time import time
import sys
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

from gcn.utils import *
from gcn.dsgcn import DoublyStochasticGCN
from gcn.vrgcn import VRGCN
from gcn.mlp import NeighbourMLP
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
flags.DEFINE_integer('num_reps', 1, 'Number of replicas')
flags.DEFINE_integer('num_layers', 1, 'Number of layers')

L = FLAGS.num_layers

# Load data
num_data, adj, features, features1, labels, train_d, val_d, test_d = \
        load_data(FLAGS.dataset)

placeholders = {
    'adj':    [tf.sparse_placeholder(tf.float32, name='adj_%d'%l) 
               for l in range(L)],
    'fields': [tf.placeholder(tf.int32, shape=(None),name='field_%d'%l) 
               for l in range(L+1)],
    'labels': tf.placeholder(tf.float32, shape=(None, labels.shape[1]), 
              name='labels'),
    'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
    'is_training': tf.placeholder(tf.bool, shape=(), name='is_training'),
    'alpha': tf.placeholder(tf.float32, shape=(), name='alpha')
}

train_degrees   = np.array([1]*L, dtype=np.int32)
eval_sch = PyScheduler(adj, labels, L, train_degrees, placeholders)
feed_dict = eval_sch.batch(test_d)
f0 = feed_dict[placeholders['fields'][0]]
ad  = tuple_to_coo(feed_dict[placeholders['adj'][0]]).tocsr()
ad = (ad>0).astype(np.float32)
a = ad.sum(0)
a = np.sort(a)
a = a[0,::-1]
for i in range(100):
    print(a[0,i])

deg = (adj>0).astype(np.float32).sum(0)
print(ad.dot(deg[0,f0].transpose()).sum())


## Analyze 
#
#adj = (adj>0).astype(np.float32)
#degree = adj.sum(0)
#
#d2 = np.copy(degree)[0]
#d2.sort()
#d2 = d2[::-1]
#fig, ax = plt.subplots()
#ax.loglog(d2, '.')
#fig.savefig('results/degree.pdf')
#
## D_i = \sum_j 1/D_j
#deg2 = adj.dot(1.0 / degree.transpose())
#d2 = np.copy(deg2)[:,0]
#d2.sort()
#d2 = d2[::-1]
#fig, ax = plt.subplots()
#ax.loglog(d2, 'r.')
#fig.savefig('results/degree2.pdf')
#
