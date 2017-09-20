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

    def _build(self):
        raise NotImplementedError

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


    def build_input(self):
        if self.sparse_input:
            self.inputs = tf.sparse_placeholder(tf.float32, name='input')
        else:
            self.inputs = tf.placeholder(tf.float32, name='input')


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


