from gcn.layers import *
from gcn.metrics import *
from gcn.inits import *
from time import time
import scipy.sparse as sp
from gcn.utils import sparse_to_tuple, dropout
from gcn.models import GCN
import numpy as np
from history import mean_history

flags = tf.app.flags
FLAGS = flags.FLAGS


class VRGCN(GCN):
    def __init__(self, L, preprocess, placeholders, 
                 features, train_features, test_features, train_adj, test_adj,
                 **kwargs):
        super(VRGCN, self).__init__(L, preprocess, placeholders, 
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
        # Create history after each aggregation
        self.history_ph      = []
        self.history_mean_ph = []
        self.history         = []
        for i in range(self.L):
            dims = self.agg0_dim if i==0 else FLAGS.hidden1
            history = tf.Variable(tf.zeros((self.num_data, dims), dtype=np.float32), 
                                  trainable=False, name='history_{}'.format(i))
            self.history.append(history)
            
            ifield = self.placeholders['fields'][i]
            fadj   = self.placeholders['fadj'][i]
            self.history_ph     .append(tf.gather(history, ifield))
            self.history_mean_ph.append(tf.sparse_tensor_dense_matmul(fadj, history))

    def get_data(self, feed_dict, is_training):
        # Read history
        for l in range(self.L):
            ofield = feed_dict[self.placeholders['fields'][l+1]]
            fadj   = self.train_adj[ofield] if is_training else self.test_adj[ofield]
            feed_dict[self.placeholders['fadj'][l]] = sparse_to_tuple(fadj)
#            self.g_ops += fadj.nnz * self.history[l].shape[1] * 2
#            self.amt_in += feed_dict[self.history_ph[l]].size + feed_dict[self.history_mean_ph[l]].size
#        self.amt_in += feed_dict[self.inputs_ph].size
#
#        for c, l in self.layer_comp:
#            self.nn_ops += c * feed_dict[self.placeholders['fields'][l]].shape[0] * 4

    def run_one_step(self, sess, feed_dict, is_training):
        t = time()
        self.get_data(feed_dict, is_training)
        self.g_t += time() - t

        # Run
        t = time()
        if is_training:
            outs = sess.run([self.opt_op, self.loss, self.accuracy], feed_dict=feed_dict)
        else:
            outs = sess.run([self.loss, self.accuracy, self.pred], feed_dict=feed_dict)
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
        for l in range(self.L):
            ifield = self.placeholders['fields'][l]
            agg = VRAggregator(adjs[l], self.history_ph[l],
                              self.history_mean_ph[l], 
                              self.placeholders['is_training'],
                              name='agg%d'%l)
            self.aggregators.append(agg)

