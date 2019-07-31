from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
import tensorflow as tf
from tensorflow_demo.pra1.practice_fun1 import get_cifar10_data, run_model
"""
    input -> conv -> relu -> BN -> max pool -> fc -> relu -> fc -> output
"""
tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)


def complex_model(X, y ,is_training):
    MOVING_AVERAGE_DECAY = 0.9997
    BN_DECAY = MOVING_AVERAGE_DECAY
    BN_EPSILON = 0.001

    Wconv1 = tf.get_variable("Wconv1", [7, 7, 3, 32])
    bconv1 = tf.get_variable("bconv1", [32])
    z1 = tf.nn.conv2d(X, Wconv1, strides=[1, 1, 1, 1], padding='VALID') + bconv1
    a1 = tf.nn.relu(z1)
    axis = list(range(len(a1.get_shape()) - 1))
    mean, variance = tf.nn.moments(a1, axis)
    param_shape = a1.get_shape()[-1:]
    moving_mean = tf.get_variable("moving_mean", param_shape, initializer=tf.zeros_initializer, trainable=False)
    moving_variance = tf.get_variable("moving_average", param_shape, initializer=tf.ones_initializer, trainable=False)
    beta = tf.get_variable("beta", param_shape, initializer=tf.zeros_initializer)
    gamma = tf.get_variable("gamma", param_shape, initializer=tf.ones_initializer)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance,  BN_DECAY)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_variance)
    mean, variance = control_flow_ops.cond(is_training,
                                           lambda: (mean, variance),
                                           lambda: (update_moving_mean, update_moving_variance))
    a1_bn = tf.nn.batch_normalization(a1, mean, variance, beta, gamma, BN_EPSILON)
    m1 = tf.nn.max_pool(a1_bn, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    m1_flat = tf.reshape(m1, [-1, 5408])
    W1 = tf.get_variable("W1", [5408, 1024])
    b1 = tf.get_variable("b1", [1024])
    z2 = tf.matmul(m1_flat, W1) + b1
    a2 = tf.nn.relu(z2)
    W2 = tf.get_variable("W2", [1024, 10])
    b2 = tf.get_variable("b2", [10])
    y_out = tf.matmul(a2, W2) + b2
    return y_out

X_train, y_train, X_val, y_val, X_test, y_test = get_cifar10_data()
y_out = complex_model(X, y, is_training)
total_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y, 10), logits=y_out)
mean_loss = tf.reduce_mean(total_loss)
optimizer = tf.train.RMSPropOptimizer(1e-3)
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = optimizer.minimize(mean_loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    run_model(sess, X, y, is_training, y_out, mean_loss, X_train, y_train, 1, 64, 100, train_step)
    run_model(sess, X, y, is_training, y_out, mean_loss, X_test, y_test, 1, 64)
