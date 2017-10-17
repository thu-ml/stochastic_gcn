import tensorflow as tf
import numpy as np


def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    return tf.get_variable(name=name, shape=shape, dtype=tf.float32,
                           initializer=tf.initializers.random_uniform(minval=-scale, maxval=scale))

def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    return tf.get_variable(name=name, shape=shape, dtype=tf.float32)


def zeros(shape, name=None):
    """All zeros."""
    return tf.get_variable(name=name, shape=shape, dtype=tf.float32, 
                           initializer=tf.zeros_initializer())


def ones(shape, name=None):
    """All ones."""
    return tf.get_variable(name=name, shape=shape, dtype=tf.float32, 
                           initializer=tf.ones_initializer())
