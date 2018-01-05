from gcn.layers import *
from gcn.metrics import *
from gcn.inits import *
from time import time
import scipy.sparse as sp
from gcn.utils import sparse_to_tuple, tuple_to_coo, np_dropout, np_sparse_dropout
from gcn.models import GCN
import numpy as np
from history import slice, dense_slice

flags = tf.app.flags
FLAGS = flags.FLAGS


class PlainGCN(GCN):
    def __init__(self, L, preprocess, placeholders, 
                 features, nbr_features, adj, cvd,
                 **kwargs):
        super(PlainGCN, self).__init__(L, preprocess, placeholders, 
                                    features, nbr_features,
                                    adj, cvd, **kwargs)

    def _build_history(self):
        self.history = []

    def get_data(self, feed_dict):
        input = self.features 
        f0    = feed_dict[self.placeholders['fields'][0]]
        dropout = feed_dict.get(self.placeholders['dropout'], 0.0)
        if self.sparse_input:
            input = slice(input, f0)
            if FLAGS.reverse:
                input = sparse_to_tuple(np_sparse_dropout(tuple_to_coo(input), 1-dropout))
        else:
            input = dense_slice(input, f0)
            if FLAGS.reverse:
                input = np_dropout(input, 1-dropout)
            #input = input[f0,:]
        feed_dict[self.inputs_ph] = input
        
        for l in range(self.L):
            dim = self.agg0_dim if l==0 else FLAGS.hidden1
            adj = feed_dict[self.placeholders['adj'][l]][0]
            self.g_ops += adj.shape[0] * dim * 4
            self.adj_sizes[l] += adj.shape[0]
            self.amt_data += adj.shape[0]
        for l in range(self.L+1):
            self.field_sizes[l] += feed_dict[self.placeholders['fields'][l]].size

        for c, l in self.layer_comp:
            self.nn_ops += c * feed_dict[self.placeholders['fields'][l]].size * 4

    def run_one_step(self, sess, feed_dict):
        t = time()
        self.get_data(feed_dict)
        self.g_t += time() - t

        # Run
        t = time()
        if self.is_training:
            outs = sess.run([self.train_op, self.loss, self.accuracy], feed_dict=feed_dict)
        else:
            outs, _ = sess.run([[self.loss, self.accuracy, self.pred], self.test_op], feed_dict=feed_dict)
        self.run_t += time() - t

        return outs


    def get_pred_and_grad(self, sess, feed_dict):
        self.get_data(feed_dict)

        # Run
        pred, grads = sess.run([self.pred, self.grads], 
                               feed_dict=feed_dict)

        return pred, grads


    def _build_aggregators(self):
        adjs   = self.placeholders['adj']
        for l in range(self.L):
            self.aggregators.append(
                    PlainAggregator(adjs[l], name='agg%d'%l))

