from gcn.inits import *
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.distributions import Normal

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


def MyLayerNorm(x):
    mean, variance = tf.nn.moments(x, axes=[1], keep_dims=True)
    
    offset = zeros([1, x.get_shape()[1]], name='offset')
    scale  = ones([1, x.get_shape()[1]], name='scale')
    return tf.nn.batch_normalization(x, mean, variance, offset, scale, 1e-9)


def MyLayerNorm2(x, offset, scale):
    mean, variance = tf.nn.moments(x, axes=[1], keep_dims=True)
    return tf.nn.batch_normalization(x, mean, variance, offset, scale, 1e-9)


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
        self.log_values = [] # TODO hack

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        self.log_values.append(x) # TODO hack

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)


        self.log_values.append(output) # TODO hack

        with tf.variable_scope(self.name + '_vars'):
            if self.norm:
                #output = layers.layer_norm(output)
                output = MyLayerNorm(output)
                self.log_values.append(output) # TODO hack

        output = self.act(output)
        self.log_values.append(output) # TODO hack
        return output


class DetDropoutFC(Layer):
    """X->Dropout->Linear->LayerNorm->ReLU->mean"""
    def __init__(self, keep_prob, input_dim, output_dim, placeholders, sparse_inputs=False,
                 norm=True, **kwargs):
        # TODO sparse inputs
        super(DetDropoutFC, self).__init__(**kwargs)
        self.sparse_inputs = sparse_inputs
        self.norm = norm
        self.keep_prob = keep_prob
        self.normal = Normal(0.0, 1.0)
        self.log_values = []

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if norm:
                self.vars['offset'] = zeros([1, output_dim], name='offset')
                self.vars['scale']  = ones ([1, output_dim], name='scale')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        # Dropout
        p   = self.keep_prob
        if isinstance(inputs, tuple):
            mu, var = inputs
            mu2 = tf.square(mu)
            var = (var+mu2) / p - mu2
        else:
            mu = inputs
            var = (1-p)/p * tf.square(inputs)

        self.log_values.append((mu, var))

        # Linear
        mu  = dot(mu, self.vars['weights'], sparse=self.sparse_inputs)
        var = dot(var, tf.square(self.vars['weights']), sparse=self.sparse_inputs) * 1.2 # TODO hack
        self.log_values.append((mu, var))

        # Norm
        if self.norm:
            mean, variance = tf.nn.moments(mu, axes=[1], keep_dims=True)
            mu  = tf.nn.batch_normalization(mu, mean, variance, self.vars['offset'], self.vars['scale'], 1e-10)
            var = var * (tf.square(self.vars['scale']) / variance)
            self.log_values.append((mu, var))

        # ReLU
        sigma = tf.sqrt(var)
        alpha = -mu / sigma
        phi   = self.normal.prob(alpha)
        Phi   = self.normal.cdf(alpha)
        Z     = self.normal.cdf(-alpha) + 1e-10
        phiZ  = phi/Z
        
        m     = mu + sigma * phiZ
        mu    = Z * m
        #var   = Z * Phi * tf.square(m)        # TODO approximation
        var   = tf.nn.relu(var * (1 + alpha*phiZ - tf.square(phiZ))) + 1e-10
        var   = Z * var + Z*Phi*tf.square(mu)
        self.log_values.append((mu, var))
        return mu, var


class GatherAggregator(Layer):
    def __init__(self, field, **kwargs):
        super(GatherAggregator, self).__init__(**kwargs)
        self.field = field

    def _call(self, inputs):
        return tf.gather(inputs, self.field)


class PlainAggregator(Layer):
    # H -> Z=AH
    # Z = concat(adj * inputs, inputs)
    def __init__(self, adj, **kwargs):
        super(PlainAggregator, self).__init__(**kwargs)

        self.adj    = adj


    def _call(self, inputs):
        ofield_size = self.adj.dense_shape[0]

        #if FLAGS.cv2:
        #    inputs = inputs[0]
        #    a_self      = inputs[:tf.cast(ofield_size, tf.int32)]

        #    # ofield * d
        #    a_neighbour = dot(self.adj, inputs, sparse=True)
        #    if FLAGS.normalization == 'gcn':
        #        return a_neighbour
        #    else:
        #        return tf.concat((a_self, a_neighbour), axis=1)
        if isinstance(inputs, tuple):
            mu, var = inputs
            mu_self      = mu[:tf.cast(ofield_size, tf.int32)]
            var_self     = var[:tf.cast(ofield_size, tf.int32)]

            mu_neighbour  = dot(self.adj, mu, sparse=True)
            var_neighbour = dot(tf.square(self.adj), var, sparse=True)

            if FLAGS.normalization == 'gcn':
                return (mu_neighbour, var_neighbour)
            else:
                return (tf.concat((mu_self, mu_neighbour), axis=1),
                        tf.concat((var_self, var_neighbour), axis=1))
        else:
            a_self      = inputs[:tf.cast(ofield_size, tf.int32)]

            # ofield * d
            a_neighbour = dot(self.adj, inputs, sparse=True)
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
    def __init__(self, adj, fadj, madj, ifield, ffield, history, scale, cvd, **kwargs):
        super(VRAggregator, self).__init__(**kwargs)

        self.adj           = adj
        self.fadj          = fadj
        self.madj          = madj
        self.ifield        = ifield
        self.ffield        = ffield
        self.history       = history
        self.scale         = scale
        self.cvd           = cvd

    def _call(self, inputs):
        ofield_size = tf.cast(self.adj.dense_shape[0], tf.int32)

        if self.cvd:
            h, mu  = inputs
            #h = inputs
            h_self  = h[:ofield_size]
            mu_self  = mu[:ofield_size]

            mu_small = tf.gather(self.history[0], self.ifield)
            mu_large = tf.gather(self.history[0], self.ffield)
            z        = h - mu
            delta_mu = mu - mu_small
            mu_mean  = dot(self.fadj, mu_large, sparse=True)

            mu_neighbour = dot(self.adj, delta_mu, sparse=True) + mu_mean
            h_neighbour = dot(self.adj, z, sparse=True) * tf.expand_dims(self.scale, 1) + mu_neighbour
                          
            self.new_history = [mu]

            if FLAGS.normalization == 'gcn':
                return (h_neighbour, mu_neighbour)
            else:
                return (tf.concat((h_self, h_neighbour), axis=1),
                        tf.concat((mu_self, mu_neighbour), axis=1))
        elif isinstance(inputs, tuple):
            mu, var  = inputs
            mu_self  = mu[:ofield_size]
            var_self = var[:ofield_size]

            mu_history, var_history = self.history

            delta_mu = mu - tf.gather(mu_history, self.ifield)
            mu_bar   = tf.gather(mu_history, self.ffield)

            sigma       = tf.sqrt(var)
            sigma_bar   = tf.sqrt(tf.gather(var_history, self.ifield))
            delta_sigma = sigma - sigma_bar
            var_bar     = tf.gather(var_history, self.ffield)
            msigma      = delta_sigma * sigma_bar

            mu_neighbour  = dot(self.adj, delta_mu, sparse=True) + dot(self.fadj, mu_bar, sparse=True)
            var_neighbour = dot(tf.square(self.adj), tf.square(delta_sigma), sparse=True) + \
                            dot(tf.square(self.fadj), var_bar, sparse=True) + \
                            2 * dot(self.madj, msigma, sparse=True)
            #var_neighbour = tf.Print(var_neighbour, [tf.reduce_min(var_neighbour)])
            var_neighbour = tf.nn.relu(var_neighbour) + 1e-10

            self.new_history = (mu, var)

            if FLAGS.normalization == 'gcn':
                return (mu_neighbour, var_neighbour)
            else:
                return (tf.concat((mu_self, mu_neighbour), axis=1),
                        tf.concat((var_self, var_neighbour), axis=1))
        else:
            history = self.history[0]
            a_self      = inputs[:ofield_size]
            a_neighbour_current = dot(self.adj, inputs, sparse=True)
            a_neighbour_history = dot(self.adj, tf.gather(history, self.ifield), sparse=True)
            a_history_mean      = dot(self.fadj, tf.gather(history, self.ffield), sparse=True)
            a_neighbour         = a_neighbour_current - a_neighbour_history + a_history_mean
            self.new_history    = [inputs]

            if FLAGS.normalization == 'gcn':
                return a_neighbour
            else:
                return tf.concat((a_self, a_neighbour), axis=1)


class AugmentedDropoutDense(Layer):
    """Dense layer."""
    def __init__(self, keep_prob, input_dim, output_dim, sparse_inputs=False,
                 act=tf.nn.relu, norm=True, **kwargs):
        super(AugmentedDropoutDense, self).__init__(**kwargs)

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.keep_prob = keep_prob
        self.norm = norm

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if norm:
                self.vars['offset'] = zeros([1, output_dim], name='offset')
                self.vars['scale']  = ones ([1, output_dim], name='scale')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        print('AugmentedDropoutDense')
        if isinstance(inputs, tuple):
            x, mu = inputs
        else:
            x, mu = inputs, inputs

        if isinstance(inputs, tf.SparseTensor):
            x = sparse_dropout(x, self.keep_prob)
        else:
            x = tf.nn.dropout(x, self.keep_prob)
        #x = inputs
        # Dropout

        # transform
        x  = dot(x, self.vars['weights'], sparse=self.sparse_inputs)
        mu = dot(mu, self.vars['weights'], sparse=self.sparse_inputs)

        with tf.variable_scope(self.name + '_vars'):
            if self.norm:
                #x  = MyLayerNorm(x)
                x  = MyLayerNorm2(x,  self.vars['offset'], self.vars['scale'])
                mu = MyLayerNorm2(mu, self.vars['offset'], self.vars['scale'])

        x  = self.act(x)
        mu = self.act(mu)
        return (x, tf.stop_gradient(mu))


class Dropout(Layer):
    def __init__(self, keep_prob, cvd, **kwargs):
        super(Dropout, self).__init__(**kwargs)

        self.keep_prob   = keep_prob
        self.cvd         = cvd

    def _call(self, inputs):
        if self.cvd and isinstance(inputs, tuple):
            h, mu = inputs
            return tf.nn.dropout(h, self.keep_prob)
        if isinstance(inputs, tuple):
            mu, var = inputs
            x = mu + tf.random_normal(tf.shape(var)) * tf.sqrt(var + 1e-10)
            return tf.nn.dropout(x, self.keep_prob)
        elif isinstance(inputs, tf.SparseTensor):
            return sparse_dropout(inputs, self.keep_prob)
        else:
            return tf.nn.dropout(inputs, self.keep_prob)


class DenoisingDropout(Layer):
    def __init__(self, keep_prob, **kwargs):
        super(Dropout, self).__init__(**kwargs)

        self.keep_prob   = keep_prob

        with tf.variable_scope(self.name + '_vars'):
            self.vars['encoder'] = glorot([input_dim, output_dim],
                                          name='encoder')
            self.vars['decoder'] = glorot([output_dim, input_dim],
                                          name='decoder')

    def _call(self, inputs):
        dropout_input = tf.nn.dropout(inputs, self.keep_prob)
        code          = dot(dropout_input, self.vars['encoder'])
        recons        = dot(code, self.vars['decoder'])
        loss          = tf.reduce_mean(tf.square(inputs-recons)) * FLAGS.denoise_factor
        return dropout_input


class Normalize(Layer):
    def __init__(self, **kwargs):
        super(Normalize, self).__init__(**kwargs)

    def _call(self, inputs):
        return tf.nn.l2_normalize(inputs, 1)
