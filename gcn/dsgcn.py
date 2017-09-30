from gcn.layers import *
from gcn.metrics import *
from gcn.inits import *
from time import time
import scipy.sparse as sp
from gcn.utils import sparse_to_tuple, dropout
from gcn.models import GCN
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS


class DoublyStochasticGCN(GCN):
    def __init__(self, data_per_fold, L, preprocess, placeholders, 
                 features, adj, 
                 **kwargs):
        super(DoublyStochasticGCN, self).__init__(data_per_fold, 
                L, preprocess, placeholders, features, adj, **kwargs)

    def _build_history(self):
        # Create history after each aggregation
        self.history_ph = []
        self.history    = []
        for i in range(self.L):
            dims = self.agg0_dim if i==0 else FLAGS.hidden1
            self.history_ph.append(tf.placeholder(tf.float32, name='agg{}_ph'.format(i)))
            self.history.append(np.zeros((self.num_data, dims), dtype=np.float32))

    def get_data(self, feed_dict, is_training):
        ids = feed_dict[self.placeholders['fields'][0]]
        if self.sparse_input:
            feed_dict[self.inputs_ph] = sparse_to_tuple(self.features[ids])
        else:
            feed_dict[self.inputs_ph] = self.features[ids]

        # Read history
        for l in range(self.L):
            field = feed_dict[self.placeholders['fields'][l+1]]
            feed_dict[self.history_ph[l]] = self.history[l][field]

    def run_one_step(self, sess, feed_dict, is_training):
        self.get_data(feed_dict, is_training)

        # Run
        if is_training:
            outs, hist, values = sess.run([[self.opt_op, self.loss, self.accuracy], 
                                           self.history_ops, self.average_get_ops],
                                  feed_dict=feed_dict)
        else:
            outs, hist, values = sess.run([[self.loss, self.accuracy, self.pred],
                                           self.history_ops, self.average_get_ops],
                                     feed_dict=feed_dict)
        self.average_model(values)

        # Write history 
        for l in range(self.L):
            field = feed_dict[self.placeholders['fields'][l+1]]
            self.history[l][field] = hist[l]

        return outs

    def _build_aggregators(self):
        adjs   = self.placeholders['adj']
        alpha  = self.placeholders['alpha']
        for l in range(self.L):
            self.aggregators.append(
                    EMAAggregator(adjs[l], alpha, 
                                  self.history_ph[l], name='agg%d'%l))

