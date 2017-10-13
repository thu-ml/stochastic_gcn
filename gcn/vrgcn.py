from gcn.layers import *
from gcn.metrics import *
from gcn.inits import *
from time import time
import scipy.sparse as sp
from gcn.utils import sparse_to_tuple, dropout
from gcn.models import GCN
import numpy as np
from history import slice

flags = tf.app.flags
FLAGS = flags.FLAGS


class VRGCN(GCN):
    def __init__(self, L, preprocess, placeholders, 
                 features, train_features, test_features, train_adj, test_adj,
                 **kwargs):
        super(VRGCN, self).__init__(L, preprocess, placeholders, 
                                    features, train_features, test_features,
                                    train_adj, test_adj, **kwargs)

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
            print('History size = {} GB'.format(self.num_data*dims*4/1024.0/1024.0/1024.0))
            
            ifield = self.placeholders['fields'][i]
            fadj   = self.placeholders['fadj'][i]
            self.history_ph     .append(tf.gather(history, ifield))
            self.history_mean_ph.append(tf.sparse_tensor_dense_matmul(fadj, history))

    def get_data(self, feed_dict, is_training):
        # Read history
        for l in range(self.L):
            ofield = feed_dict[self.placeholders['fields'][l+1]]
            #fadj   = self.train_adj[ofield] if is_training else self.test_adj[ofield]
            fadj = slice(self.train_adj, ofield) if is_training else slice(self.test_adj, ofield)
            adj = feed_dict[self.placeholders['adj'][l]][0]
            feed_dict[self.placeholders['fadj'][l]] = sparse_to_tuple(fadj)

            dim = self.agg0_dim if l==0 else FLAGS.hidden1
            self.g_ops += fadj.nnz * dim * 2
            self.g_ops += adj.shape[0] * dim * 2
            self.adj_sizes[l] += adj.shape[0]
            self.fadj_sizes[l] += fadj.nnz
        for l in range(self.L+1):
            self.field_sizes[l] += feed_dict[self.placeholders['fields'][l]].size
        for c, l in self.layer_comp:
            self.nn_ops += c * feed_dict[self.placeholders['fields'][l]].size * 4

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
        for l in range(self.L):
            ifield = self.placeholders['fields'][l]
            agg = VRAggregator(adjs[l], self.history_ph[l],
                              self.history_mean_ph[l], 
                              self.placeholders['is_training'],
                              name='agg%d'%l)
            self.aggregators.append(agg)

