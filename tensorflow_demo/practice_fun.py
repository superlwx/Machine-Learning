from keras.datasets import cifar10
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math


"""
Tensorflow 框架的简单练习

"""


def get_cifar10_data(num_training=49000, num_validation=1000, num_test=10000):
    """
    获取cifar数据，把数据划分为训练集，验证集，测试集
    :param num_training:
    :param num_validation:
    :param num_test:
    :return: (X_train, y_train, X_val, y_val, X_test, y_test)
    """

    (X_train, y_train),(X_test, y_test) = cifar10.load_data()

    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask].astype('float64')
    y_val = y_train[mask]

    mask = range(num_training)
    X_train = X_train[mask].astype('float64')
    y_train = y_train[mask]
    mean = np.mean(X_train, axis=0)  # 数据零均值化
    X_test = X_test.astype('float64')
    X_train -= mean
    X_val -= mean
    X_test -= mean
    return X_train, y_train, X_val, y_val, X_test, y_test


def simple_model(X, y):
    """
    实现 input -> conv -> relu -> fully_connected -> output 的正向传播
    :param X:
    :param y:
    :return:
    """
    Wconv1 = tf.compat.v1.get_variable("Wconv1", shape=[7, 7, 3, 32])
    bconv1 = tf.compat.v1.get_variable("bconv1", shape=[32])
    W1 = tf.compat.v1.get_variable("W1", [5408, 10])
    b1 = tf.compat.v1.get_variable("b1", [10])
    a1 = tf.compat.v1.nn.conv2d(X, Wconv1, strides=[1, 2, 2, 1], padding="VALID") + bconv1
    h1 = tf.nn.relu(a1)
    h1_flat = tf.reshape(h1, [-1, 5408])
    y_out = tf.matmul(h1_flat, W1) + b1
    return y_out


def run_model(session, X, y, is_training, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    correct_predicion = tf.equal(tf.math.argmax(predict, axis=1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_predicion, tf.float32))
    train_indices = np.arange(Xd.shape[0])
    np.random.shuffle(train_indices)
    training_now = training is not None
    variables = [loss_val, correct_predicion, accuracy]
    if training_now:
        variables.append(training)
    iter_cnt = 0
    print(len(variables))
    for e in range(epochs):
        correct = 0
        losses = []
        for i in range(math.ceil(Xd.shape[0] / batch_size)):
            start_idx = (i * batch_size) % Xd.shape[0]
            idx = train_indices[start_idx: start_idx + batch_size]
            actual_batch_size = yd[idx].shape[0]
            if len(variables) == 4:
                loss, corr, acc, _ = session.run(variables, feed_dict={X: Xd[idx],
                                                                       y: np.squeeze(yd[idx]),
                                                                       is_training: training_now})
            else:
                loss, corr, acc = session.run(variables, feed_dict={X: Xd[idx],
                                                                    y: np.squeeze(yd[idx]),
                                                                    is_training: training_now})
            losses.append(loss * actual_batch_size)
            correct += np.sum(corr)
            if training_now and (iter_cnt % print_every == 0):
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"
                      .format(iter_cnt, loss, acc))
            iter_cnt += 1
        total_correct = correct / Xd.shape[0]
        total_loss = np.sum(losses) / Xd.shape[0]
        print("Epoch {2}, Overall loss={0:.3g} and accuracy of {1:.3g}"
              .format(total_loss, total_correct, e + 1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title("Epoch {} Loss".format(e + 1))
            plt.xlabel("minibatch number")
            plt.ylabel("minibatch loss")
            plt.show()

    return total_loss, total_correct


