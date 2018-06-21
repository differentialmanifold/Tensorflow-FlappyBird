import tensorflow as tf


def mean_squared_error(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true), name='loss')


def mean_softmax_cross_entropy(y_true, y_pred):
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_true, labels=y_pred)
    return tf.reduce_mean(entropy, name='loss')


identify = {'mean_squared_error': mean_squared_error,
            'mean_softmax_cross_entropy': mean_softmax_cross_entropy}
