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


class PlainGCN(GCN):
    def __init__(self, L, preprocess, placeholders, 
                 features, train_features, test_features, train_adj, test_adj,
                 **kwargs):
        super(PlainGCN, self).__init__(L, preprocess, placeholders, 
                                    features, train_features, test_features,
                                    train_adj, test_adj, **kwargs)

        self.run_t = 0
        self.g_t   = 0
        self.h_t   = 0
        self.g_ops = 0
        self.nn_ops = 0
        self.amt_in = 0
        self.amt_out = 0

    def _build_history(self):
        pass

    def get_data(self, feed_dict, is_training):
        pass

    def run_one_step(self, sess, feed_dict, is_training):
        t = time()
        self.get_data(feed_dict, is_training)
        self.g_t += time() - t

        # Run
        t = time()
        if is_training:
            outs = sess.run([self.train_op, self.loss, self.accuracy], feed_dict=feed_dict)
        else:
            outs, _ = sess.run([[self.loss, self.accuracy, self.pred], self.test_op], feed_dict=feed_dict)
        self.run_t += time() - t

        return outs


    def get_pred_and_grad(self, sess, feed_dict, is_training):
        self.get_data(feed_dict, is_training)

        # Run
        pred, grads = sess.run([self.pred, self.grads], 
                               feed_dict=feed_dict)

        return pred, grads


    def _build_aggregators(self):
        adjs   = self.placeholders['adj']
        alpha  = self.placeholders['alpha']
        for l in range(self.L):
            self.aggregators.append(
                    PlainAggregator(adjs[l], name='agg%d'%l))

