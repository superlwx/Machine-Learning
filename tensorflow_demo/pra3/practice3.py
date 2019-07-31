import tensorflow as tf
from tensorflow_demo.pra1.practice_fun1 import run_model, get_cifar10_data
"""
    tensorflow.contrib.layers的使用
"""


def my_model(X, y, is_training):
    def conv_relu_pool(X, num_filter=32, conv_stride=1,
                       kernel_size=[3, 3], pool_size=[2, 2], pool_stride=2):
        conv1 = tf.layers.conv2d(inputs=X, filters=num_filter, kernel_size=kernel_size,
                                 stride=conv_stride, padding="same", activation_fn=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=pool_size, strides=pool_stride)
        return pool1

    def conv_relu_conv_relu_pool(X, num_filter1=32, num_filter2=32, conv_stride=1,
                                 kernel_size=[5, 5], pool_size=[2, 2], pool_stride=2):
        conv1 = tf.layers.conv2d(inputs=X, filters=num_filter1, kernel_size=kernel_size,
                                 strides=conv_stride, padding="same", activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=num_filter2, kernel_size=kernel_size,
                                 strides=conv_stride, padding="same", activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=pool_size,
                                        strides=pool_stride)
        return pool1

    def affine(X, num_units, act):
        return tf.layers.dense(X, num_units, act)

    def batchnorm_relu_conv(X, is_training=True, kernel_size=[3, 3], conv_stride=2,
                            num_filter=32):
        bn1 = tf.layers.batch_normalization(X, training=is_training)
        act1 = tf.nn.relu(bn1)
        conv1 = tf.layers.conv2d(inputs=act1, filters=num_filter, kernel_size=kernel_size,
                                 strides=conv_stride, padding="same", activation=None,
                                 kernel_initializer=tf.variance_scaling_initializer)
        return conv1

    N = 3  # 3 conv blocks
    M = 1  # 1 affine block
    conv = tf.layers.conv2d(inputs=X, filters=64, kernel_size=[3, 3], strides=1,
                            padding="same", activation=None, kernel_initializer=tf.variance_scaling_initializer)
    for i in range(N):
        print(conv.get_shape())
        conv = batchnorm_relu_conv(conv, is_training)

    print(conv.get_shape())
    conv = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2)
    fc = tf.layers.flatten(conv)
    print(fc.get_shape())

    for i in range(M):
        fc = affine(fc, 100, tf.nn.relu)
        fc = tf.nn.dropout(fc, 0.9)
    print(fc.get_shape())
    fc = affine(fc, 10, None)
    print(fc.get_shape())
    return fc


tf.reset_default_graph()
X_train, y_train, X_val, y_val, X_test, y_test = get_cifar10_data()
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)
y_out = my_model(X, y, is_training)
total_loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_out, labels=tf.one_hot(y, 10))
mean_loss = tf.reduce_mean(total_loss)

gloal_step = tf.Variable(0, trainable=False, name="Global_step")
start_learning_rate = 1e-2
learing_rate = tf.train.exponential_decay(start_learning_rate, gloal_step, 750, 0.96, staircase=True)
optimizer = tf.train.AdamOptimizer(learing_rate)

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = optimizer.minimize(mean_loss)
print([x.name for x in tf.global_variables()])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    run_model(sess, X, y, is_training, y_out, mean_loss, X_train, y_train, 5, 64, 100, train_step)
    run_model(sess, X, y, is_training, y_out, mean_loss, X_test, y_test, 1, 64)

