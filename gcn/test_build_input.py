from models import build_input
import tensorflow as tf
import numpy as np
import scipy.sparse as sp

# features = sp.csr_matrix(np.array(
#     [[0, 1, 0],
#      [2, 0, 3],
#      [4, 5, 6]]))
features = np.array(
    [[0, 1, 0],
     [2, 0, 3],
     [4, 5, 6]])

update_ops = []
feed_dict = {}
t = build_input(features, 'test', update_ops, feed_dict)

mat = tf.placeholder(tf.float32)
# result = tf.sparse_tensor_dense_matmul(t, mat)
result = tf.matmul(t, mat)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(update_ops, feed_dict=feed_dict)
    print(sess.run(t))
    print(sess.run(result, feed_dict={mat: np.array([[1], [2], [3]])}))
