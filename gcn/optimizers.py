import tensorflow as tf
from utils import least_squares, least_squares_A, hinge_loss


def optimize_w(W, Z, A, decay):
    # W = argmin ||Z - AW||^2 + decay * ||W||^2
    new_W = least_squares(A, Z, decay)
    return tf.assign(W, new_W)


def optimize_a(A, H, Z, W, a_penalty, z_penalty):
    # A = argmin gamma * ||A - H||^2 + beta * ||AW - Z||^2
    new_A = least_squares_A(H, Z, W, a_penalty, z_penalty)
    return tf.assign(A, new_A)


def optimize_z(Z, A, C, a_penalty, z_penalty):
    # z = argmin_Z gamma * (a-h(z))^2 + beta * (z - c)^2
    best_z_lt_zero = tf.minimum(0.0, C)
    best_z_gt_zero = tf.maximum(0.0, (a_penalty*A+z_penalty*C)/(a_penalty+z_penalty))
    L_lt_zero      = a_penalty * tf.square(A) + z_penalty * tf.square(best_z_lt_zero - C)
    L_gt_zero      = a_penalty * tf.square(A - best_z_gt_zero) + \
                     z_penalty * tf.square(best_z_gt_zero - C)
    L_zero         = a_penalty * tf.square(A) + z_penalty * tf.square(C)
    best_z         = tf.where(L_lt_zero < L_gt_zero, best_z_lt_zero, best_z_gt_zero)
    return tf.assign(Z, best_z)


def optimize_last_z(Z, Y, M, C, z_penalty):     
    # z = argmin_Z hinge(z, y) + z*m + 0.5 * beta * (z-c)^2
    Y2 = 2 * Y - 1
    z_gt_one = (z_penalty*C - M) / z_penalty
    z_gt_one = tf.where(z_gt_one*Y2 > 1, z_gt_one, Y2)
    z_lt_one = (z_penalty*C - M + Y2) / z_penalty
    z_lt_one = tf.where(z_lt_one*Y2 < 1, z_lt_one, Y2)
    get_loss = lambda Z: hinge_loss(Y, Z)+tf.reduce_sum(Z*M)+0.5*z_penalty*tf.square(Z-C)
    L_gt_one = get_loss(z_gt_one)
    L_lt_one = get_loss(z_lt_one)
    L_one    = get_loss(Y2)
    best_z   = tf.where(L_gt_one < L_lt_one, z_gt_one, z_lt_one)
    # best_z   = tf.Print(best_z, [tf.reduce_max(L_gt_one-L_one), tf.reduce_max(L_lt_one-L_one), z_gt_one, z_lt_one, best_z])
    return tf.assign(Z, best_z)


def optimize_multiplier(multiplier, Z, C, z_penalty):
    new_multiplier = multiplier + z_penalty * (Z - C)
    return tf.assign(multiplier, new_multiplier)


def test_a():
    # A: N*K
    # H: N*K
    # Z: N*D
    # W: K*D
    N = 100 
    K = 50
    D = 20
    gamma = 10
    beta = 1

    A = tf.Variable(tf.random_normal((N, K)) * 0.01)
    H = tf.Variable(tf.random_uniform((N, K)))
    Z = tf.Variable(tf.random_uniform((N, D)))
    W = tf.Variable(tf.random_uniform((K, D)))
    
    loss = gamma * tf.nn.l2_loss(A - H) + beta * tf.nn.l2_loss(tf.matmul(A, W) - Z)
    g_magnitude = tf.nn.l2_loss(tf.gradients(loss, [A])[0])
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
    gd_train = optimizer.minimize(loss, var_list=[A])
    solve = optimize_a(A, H, Z, W, gamma, beta)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(100):
            _, l, g = sess.run([gd_train, loss, g_magnitude])
            print('Loss = {}, gradient = {}'.format(l, g))
        sess.run(solve)
        l, g = sess.run([loss, g_magnitude])
        print('Loss = {}, gradient = {}'.format(l, g))


def test_z():
    N = 100
    D = 50
    gamma = 10
    beta = 1

    Z = tf.Variable(tf.random_normal((N, D)) * 0.01)
    A = tf.Variable(tf.random_normal((N, D)))
    C = tf.Variable(tf.random_normal((N, D)))

    loss = gamma * tf.nn.l2_loss(A - tf.nn.relu(Z)) + beta * tf.nn.l2_loss(Z - C)
    g_magnitude = tf.nn.l2_loss(tf.gradients(loss, [Z])[0])
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
    gd_train = optimizer.minimize(loss, var_list=[Z])
    solve = optimize_z(Z, A, C, gamma, beta)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(100):
            _, l, g = sess.run([gd_train, loss, g_magnitude])
            print('Loss = {}, gradient = {}'.format(l, g))
        sess.run(solve)
        l, g = sess.run([loss, g_magnitude])
        print('---------Loss = {}, gradient = {}-----------'.format(l, g))
        for epoch in range(100):
            _, l, g = sess.run([gd_train, loss, g_magnitude])
            print('Loss = {}, gradient = {}'.format(l, g))


def test_last_z():
    N = 100
    D = 50
    beta = 1

    Z = tf.Variable(tf.random_normal((N, D)) * 0.01)
    Y = tf.Variable(tf.cast(tf.random_uniform((N, D)) > 0.5, tf.float32))
    C = tf.Variable(tf.random_uniform((N, D)))
    M = tf.Variable(tf.random_uniform((N, D)))

    loss = tf.reduce_sum(hinge_loss(Y, Z)) + tf.reduce_sum(Z*M) + beta*tf.nn.l2_loss(Z-C)
    g_magnitude = tf.nn.l2_loss(tf.gradients(loss, [Z])[0])
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
    gd_train = optimizer.minimize(loss, var_list=[Z])
    solve = optimize_last_z(Z, Y, M, C, beta)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(100):
            _, l, g = sess.run([gd_train, loss, g_magnitude])
            print('Loss = {}, gradient = {}'.format(l, g))
        sess.run(solve)
        # print(sess.run(Z))
        # print(sess.run(Y))
        # print(sess.run(C))
        # print(sess.run(M))
        l, g = sess.run([loss, g_magnitude])
        print('---------Loss = {}, gradient = {}-----------'.format(l, g))
        for epoch in range(100):
            _, l, g = sess.run([gd_train, loss, g_magnitude])
            print('Loss = {}, gradient = {}'.format(l, g))


if __name__ == '__main__':
    # test_a()
    # test_z()
    test_last_z()
    # y = tf.constant([1.0, 1.0])
    # z = tf.constant([0.5, 0.3])
    # l = tf.losses.hinge_loss(labels=y, logits=z)
    # with tf.Session() as sess:
    #     print(sess.run(l))
