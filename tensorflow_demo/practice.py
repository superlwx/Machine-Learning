import tensorflow as tf
import numpy as np
from tensorflow_demo.practice_fun import simple_model,get_cifar10_data,run_model

tf.reset_default_graph()
X = tf.compat.v1.placeholder(tf.float32, [None, 32, 32, 3], name='X')
y = tf.compat.v1.placeholder(tf.int64, [None], name='y')
is_training = tf.compat.v1.placeholder(tf.bool, name='is_training')
y_out = simple_model(X, y)
total_loss = tf.compat.v1.losses.hinge_loss(tf.one_hot(y, 10), logits=y_out)
mean_loss = tf.reduce_mean(total_loss)
optimizer = tf.compat.v1.train.AdamOptimizer(5e-4)
train_step = optimizer.minimize(mean_loss)
X_train, y_train, X_val, y_val, X_test, y_test = get_cifar10_data()
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    print("Training")
    run_model(session, X, y, is_training, y_out, mean_loss, X_train, y_train, 1, 64, 100, train_step, True)
    print("Validation")
    run_model(session, X, y, is_training, y_out, mean_loss, X_test, y_test, 1, 64)
    session.close()
