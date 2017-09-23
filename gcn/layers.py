from gcn.inits import *
import tensorflow as tf
from tensorflow.contrib import layers

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob):
    """Dropout for sparse tensors."""
    random_tensor = tf.random_uniform(tf.shape(x.values))
    dropout_mask  = random_tensor < keep_prob
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False
        self.post_updates = []

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, sparse_inputs=False,
                 act=tf.nn.relu, norm=True, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.norm = norm

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        if self.norm:
            output = layers.layer_norm(output)

        return self.act(output)


class GatherAggregator(Layer):
    def __init__(self, field, **kwargs):
        super(GatherAggregator, self).__init__(**kwargs)
        self.field = field

    def _call(self, inputs):
        return tf.gather(inputs, self.field)


class PlainAggregator(Layer):
    # H -> Z=AH
    # Z = concat(adj * inputs, inputs)
    def __init__(self, adj, ifield, ofield, **kwargs):
        super(PlainAggregator, self).__init__(**kwargs)

        self.adj    = adj
        self.ifield = ifield
        self.ofield = ofield


    def _call(self, inputs):
        # ofield * d
        a_neighbour = dot(self.adj, inputs, sparse=True)
        a_self      = inputs[:tf.shape(self.ofield)[0], :]
        if FLAGS.normalization == 'gcn':
            return a_neighbour
        else:
            return tf.concat((a_self, a_neighbour), axis=1)


class EMAAggregator(Layer):
    def __init__(self, adj, alpha, history, **kwargs):
        super(EMAAggregator, self).__init__(**kwargs)

        self.adj     = adj
        self.alpha   = alpha
        self.history = history

    def _call(self, inputs):
        ofield_size = self.adj.dense_shape[0]
        a_self      = inputs[:tf.cast(ofield_size, tf.int32)]

        a_neighbour_hat  = dot(self.adj, inputs, sparse=True)
        a_neighbour      = a_neighbour_hat * self.alpha + \
                           self.history * (1-self.alpha)
        self.new_history = a_neighbour
        if FLAGS.normalization == 'gcn':
            return a_neighbour
        else:
            return tf.concat((a_self, a_neighbour), axis=1)


class VRAggregator(Layer):
    def __init__(self, adj, history, history_mean, is_training, **kwargs):
        super(VRAggregator, self).__init__(**kwargs)

        self.adj           = adj
        self.history       = history
        self.history_mean  = history_mean
        self.is_training   = is_training

    def _call(self, inputs):
        ofield_size = self.adj.dense_shape[0]
        a_self      = inputs[:tf.cast(ofield_size, tf.int32)]
        a_neighbour_current = dot(self.adj, inputs, sparse=True)

        def training():
            a_neighbour_history = dot(self.adj, self.history, sparse=True)
            a_neighbour         = a_neighbour_current - a_neighbour_history + self.history_mean
            return a_neighbour

        def testing():
            a_neighbour         = a_neighbour_current
            return a_neighbour

        a_neighbour         = tf.cond(self.is_training, training, testing)
        self.new_history    = inputs

        if FLAGS.normalization == 'gcn':
            return a_neighbour
        else:
            return tf.concat((a_self, a_neighbour), axis=1)


class Dropout(Layer):
    def __init__(self, keep_prob, is_training, **kwargs):
        super(Dropout, self).__init__(**kwargs)

        self.keep_prob   = keep_prob
        self.is_training = is_training

    def _call(self, inputs):
        if isinstance(inputs, tf.SparseTensor):
            return sparse_dropout(inputs, self.keep_prob)
        else:
            return layers.dropout(inputs, self.keep_prob, 
                                      is_training=self.is_training)


class Normalize(Layer):
    def __init__(self, **kwargs):
        super(Normalize, self).__init__(**kwargs)

    def _call(self, inputs):
        return tf.nn.l2_normalize(inputs, 1)
