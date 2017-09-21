from gcn.layers import *
from gcn.metrics import *
from gcn.inits import *
from time import time
import scipy.sparse as sp
from gcn.utils import sparse_to_tuple
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'multitask'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None
        self.multitask = kwargs.get('multitask', False)

        self.history_ops = []

    def _build(self):
        raise NotImplementedError

    def _history(self):
        pass

    def _loss(self):
        # Weight decay loss on the first layer
        l = 0
        while len(self.layers[l].vars.values()) == 0:
            l += 1
        for var in self.layers[l].vars.values():
            print('Var')
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        if self.multitask:
            self.loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.outputs, labels=self.placeholders['labels']))
        else:
            # Cross entropy error
            self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.outputs, labels=self.placeholders['labels']))

    def _accuracy(self):
        if self.multitask:
            preds = self.outputs > 0
            labs  = self.placeholders['labels'] > 0.5
            correct_prediction = tf.equal(preds, labs)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        else:
            correct_prediction = tf.equal(tf.argmax(self.outputs, 1), 
                                          tf.argmax(self.placeholders['labels'], 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def get_ph(self, name):
        if self.sparse_input:
            return tf.sparse_placeholder(tf.float32, name=name)
        else:
            return tf.placeholder(tf.float32, name=name)


    def get_zeros(self, shape):
        if self.sparse_input:
            return np.zeros(shape, dtype=np.float32)
        else:
            return sp.csr_matrix(shape, dtype=np.float32)


    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            print('{} shape = {}'.format(layer.name, hidden.get_shape()))
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        with tf.variable_scope(self.name):
            self._history()

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = [self.optimizer.minimize(self.loss)]
        with tf.control_dependencies(self.opt_op):
            for layer in self.layers:
                for ref, indices, updates in layer.post_updates:
                    self.opt_op.append(tf.scatter_update(ref, indices, updates))

    def predict(self):
        if self.multitask:
            return tf.nn.sigmoid(self.outputs)
        else:
            return tf.nn.softmax(self.outputs)

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class GraphSAGE(Model):
    def __init__(self, L, placeholders, features, train_adj=None, test_adj=None, **kwargs):
        super(GraphSAGE, self).__init__(**kwargs)

        self.L = L

        self.sparse_input = not isinstance(features, np.ndarray)
        self.build_input()
        self.input_dim  = features.shape[1]
        self.self_features = features

        if train_adj is not None:
            # Preprocess first aggregation
            print('Preprocessing first aggregation')
            start_t = time()

            self.train_features = train_adj.dot(features)
            self.test_features  = test_adj.dot(features)

            self.preprocess   = True
            print('Finished in {} seconds.'.format(time() - start_t))
        else:
            self.preprocess = False

        self.num_data = features.shape[0]
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, 
                                                beta1=FLAGS.beta1, beta2=FLAGS.beta2)

        self.build()


    def get_data(self, feed_dict, is_training):
        if not self.preprocess:
            ids = feed_dict[self.placeholders['fields'][0]]
        else:
            ids = feed_dict[self.placeholders['fields'][1]]

        nbr_inputs = self.train_features[ids] if is_training else self.test_features[ids]
        if self.sparse_input:
            if not self.preprocess:
                inputs = sparse_to_tuple(self.self_features[ids])
            elif FLAGS.normalization=='gcn':
                inputs = sparse_to_tuple(nbr_inputs)
            else:
                inputs = sparse_to_tuple(sp.hstack((self.self_features[ids], nbr_inputs)))
        else:
            nbr_inputs = self.train_features[ids] if is_training else self.test_features[ids]
            if not self.preprocess:
                inputs = self.self_features
            elif FLAGS.normalization=='gcn':
                inputs = nbr_inputs
            else:
                inputs = np.hstack((self.self_features[ids], nbr_inputs))

        feed_dict[self.inputs] = inputs


    def _build(self):
        # Aggregate
        fields = self.placeholders['fields']
        adjs   = self.placeholders['adj']
        dim_s  = 1 if FLAGS.normalization=='gcn' else 2

        if not self.preprocess:
            self.layers.append(PlainAggregator(adjs[0], fields[0], fields[1],
                                               name='agg1'))

        for l in range(1, self.L):
            input_dim = self.input_dim if l==1 else FLAGS.hidden1
            self.layers.append(Dropout(1-self.placeholders['dropout'],
                                       self.placeholders['is_training']))
            self.layers.append(Dense(input_dim=input_dim*dim_s,
                                     output_dim=FLAGS.hidden1,
                                     placeholders=self.placeholders,
                                     act=tf.nn.relu,
                                     logging=self.logging,
                                     sparse_inputs=self.sparse_input if l==1 else False,
                                     name='dense%d'%l, norm=FLAGS.layer_norm))
            self.layers.append(PlainAggregator(adjs[l], fields[l], fields[l+1], 
                                               name='agg%d'%(l+1)))

        self.layers.append(Dropout(1-self.placeholders['dropout'],
                                   self.placeholders['is_training']))
        self.layers.append(Dense(input_dim=FLAGS.hidden1*dim_s,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 logging=self.logging,
                                 name='dense2', norm=False))

# -----------------------------------------------
class NeighbourMLP(Model):
    def __init__(self, L, placeholders, features, train_adj, test_adj, **kwargs):
        super(NeighbourMLP, self).__init__(**kwargs)

        self.L = L
        self.sparse_input = not isinstance(features, np.ndarray)
        self.build_input()

        # Preprocess aggregation
        print('Preprocessing aggregations')
        start_t = time()

        # Create all the features
        def _create_features(X, A):
            features = [X]
            print('Hops = {}'.format(FLAGS.num_hops))
            for i in range(FLAGS.num_hops):
                features.append(A.dot(features[-1]))
            return np.hstack(features)

        self.train_features = _create_features(features, train_adj)
        self.test_features  = _create_features(features, test_adj)
        self.input_dim = self.train_features.shape[1]
        print('Finished in {} seconds.'.format(time() - start_t))

        self.num_data = features.shape[0]
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, 
                                                beta1=FLAGS.beta1, beta2=FLAGS.beta2)

        self.build()


    def get_data(self, feed_dict, is_training):
        ids = feed_dict[self.placeholders['fields'][-1]]
        inputs = self.train_features[ids] if is_training else self.test_features[ids]
        if self.sparse_input:
            inputs = sparse_to_tuple(inputs)

        feed_dict[self.inputs] = inputs


    def _build(self):
        for l in range(0, self.L-1):
            input_dim = self.input_dim if l==0 else FLAGS.hidden1
            self.layers.append(Dropout(1-self.placeholders['dropout'],
                                       self.placeholders['is_training']))
            self.layers.append(Dense(input_dim=input_dim,
                                     output_dim=FLAGS.hidden1,
                                     placeholders=self.placeholders,
                                     act=tf.nn.relu,
                                     logging=self.logging,
                                     sparse_inputs=self.sparse_input if l==0 else False,
                                     name='dense%d'%l))

        self.layers.append(Dropout(1-self.placeholders['dropout'],
                                   self.placeholders['is_training']))
        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 logging=self.logging,
                                 name='dense%d'%(self.L-1)))


class DoublyStochasticGCN(Model):
    def __init__(self, L, preprocess, placeholders, 
                 features, train_adj, test_adj,
                 **kwargs):
        super(DoublyStochasticGCN, self).__init__(**kwargs)

        self.preprocess     = preprocess
        self.sparse_input   = not isinstance(features, np.ndarray)
        self.sparse_mm      = self.sparse_input
        self.inputs_ph      = self.get_ph('input')
        if not self.preprocess and self.sparse_input:
            print('Warning: we do not support sparse input without pre-processing. Converting to dense...')
            self.inputs     = tf.sparse_to_dense(self.inputs_ph.indices, 
                                                 self.inputs_ph.dense_shape,
                                                 self.inputs_ph.values)
            self.sparse_mm  = False
        else:
            self.inputs     = self.inputs_ph
        self.num_data       = features.shape[0]
        self.input_dim      = features.shape[1]
        self.self_dim       = 0 if FLAGS.normalization=='gcn' else self.input_dim

        if self.preprocess:
            self.self_features  = features[:,:self.self_dim]
            self.train_features = train_adj.dot(features)
            self.test_features  = test_adj.dot(features)
            self.L              = L-1
        else:
            self.self_features  = features
            self.train_features = np.zeros((self.num_data, 0), dtype=np.float32)
            self.test_features  = np.zeros((self.num_data, 0), dtype=np.float32)
            self.L              = L

        self.agg0_dim       = FLAGS.hidden1 if self.preprocess else self.input_dim
        self.output_dim     = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders   = placeholders

        # Create history after each aggregation
        self.history_ph = []
        self.history    = []
        for i in range(self.L):
            dims = self.agg0_dim if i==0 else FLAGS.hidden1
            self.history_ph.append(tf.placeholder(tf.float32, name='agg{}_ph'.format(i)))
            self.history.append(np.zeros((features.shape[0], dims), dtype=np.float32))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, 
                                                beta1=FLAGS.beta1, beta2=FLAGS.beta2)

        self.build()


    def get_data(self, feed_dict, is_training):
        ids        = feed_dict[self.placeholders['fields'][0]]
        nbr_inputs = self.train_features[ids] if is_training else self.test_features[ids]
        if self.sparse_input:
            inputs = sparse_to_tuple(sp.hstack((self.self_features[ids], nbr_inputs)))
        else:
            inputs = np.hstack((self.self_features[ids], nbr_inputs))
        feed_dict[self.inputs_ph] = inputs

        # Read history
        for l in range(self.L):
            field = feed_dict[self.placeholders['fields'][l+1]]
            feed_dict[self.history_ph[l]] = self.history[l][field]


    def train_one_step(self, sess, feed_dict, is_training):
        self.get_data(feed_dict, is_training)

        # Run
        outs, hist = sess.run([[self.opt_op, self.loss, self.accuracy], self.history_ops],
                              feed_dict=feed_dict)

        # Write history
        for l in range(1, self.L):
            field = feed_dict[self.placeholders['fields'][l+1]]
            self.history[l][field] = hist[l]

        return outs


    def _build(self):
        # Aggregate
        fields = self.placeholders['fields']
        adjs   = self.placeholders['adj']
        dim_s  = 1 if FLAGS.normalization=='gcn' else 2
        alpha  = self.placeholders['alpha']
        cnt    = 0

        if self.preprocess:
            self.layers.append(Dropout(1-self.placeholders['dropout'],
                                       self.placeholders['is_training']))
            self.layers.append(Dense(input_dim=self.input_dim*dim_s,
                                     output_dim=FLAGS.hidden1,
                                     placeholders=self.placeholders,
                                     act=tf.nn.relu,
                                     logging=self.logging,
                                     sparse_inputs=self.sparse_mm,
                                     name='dense0', norm=FLAGS.layer_norm))
            cnt += 1

        for l in range(self.L):
            self.layers.append(EMAAggregator(adjs[l], alpha,
                                             self.history_ph[l], name='agg%d'%l))
            self.layers.append(Dropout(1-self.placeholders['dropout'],
                                       self.placeholders['is_training']))

            name = 'dense%d' % (l+cnt)
            dim  = self.agg0_dim if l==0 else FLAGS.hidden1
            if l+1==self.L:
                output_dim, norm, act = self.output_dim, False, lambda x: x
            else:
                output_dim, norm, act = FLAGS.hidden1, FLAGS.layer_norm, tf.nn.relu

            self.layers.append(Dense(input_dim=dim*dim_s,
                                     output_dim=output_dim,
                                     placeholders=self.placeholders,
                                     act=act,
                                     logging=self.logging,
                                     name=name, norm=norm))


    def _history(self):
       self.activations.append(self.inputs)
       for layer in self.layers:
           if hasattr(layer, 'new_history'):
               self.history_ops.append(layer.new_history)

