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


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
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
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders.get('num_features_nonzero', None)

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        # if self.sparse_inputs:
        #     x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        # else:
        #     x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, support=None, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        if support is None:
            self.support = placeholders['support']
        else:
            self.support = support
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class VarianceReductedAggregator(Layer):
    # H -> Z=AH
    # Z = subsampled_support * (inputs - old[input_fields]) + support * old
    # subsampled_support: outputs_fields * input_fields matrix
    # support (A)       : outputs_fields * n            matrix
    def __init__(self, num_data, input_dim,
                 input_fields, 
                 subsampled_support, support, **kwargs):
        super(VarianceReductedAggregator, self).__init__(**kwargs)

        self.input_fields = input_fields
        self.subsampled_support = subsampled_support
        self.support = support

        with tf.variable_scope(self.name + '_vars'):
            self.old_activation = tf.Variable(tf.zeros([num_data, input_dim]), trainable=False, name='activation')


    def _call(self, inputs):
        old_activations = tf.gather(self.old_activation, self.input_fields)
        norm = lambda x: tf.sqrt(tf.reduce_sum(tf.square(x)))
        sim = tf.reduce_sum(inputs*old_activations) / norm(inputs) / norm(old_activations)
        output = dot(self.subsampled_support, inputs-old_activations, sparse=True) +\
                 dot(self.support,            self.old_activation, sparse=True)
        output = tf.Print(output, [sim])

        self.post_updates.append((self.old_activation, self.input_fields, inputs))
        return output


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
        return tf.concat((a_self, a_neighbour), axis=1)

class Dropout(Layer):
    def __init__(self, keep_prob, is_training, **kwargs):
        super(Dropout, self).__init__(**kwargs)

        self.keep_prob   = keep_prob
        self.is_training = is_training

    def _call(self, inputs):
        return layers.dropout(inputs, self.keep_prob, 
                                      is_training=self.is_training)

