import tensorflow as tf
import os

stddev = 0.1


def conv_layer(
        inputs,
        out_channels,
        kernel_size,
        name,
        stride=1):
    with tf.variable_scope(name):
        in_channels = inputs.shape[-1]
        filters = kernel_size + [in_channels] + [out_channels]
        kernel = tf.get_variable('kernels', filters,
                                 initializer=tf.truncated_normal_initializer(stddev=stddev))
        biases = tf.get_variable('biases', [out_channels],
                                 initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(inputs, kernel, strides=[1, stride, stride, 1], padding='SAME')
        return tf.nn.relu(conv + biases, name='relu')


def max_pool(inputs, pool_size, stride, name):
    with tf.name_scope(name):
        ksize = [1] + pool_size + [1]
        strides = [1] + [stride, stride] + [1]
        return tf.nn.max_pool(inputs, ksize=ksize, strides=strides, padding='SAME', name='max_pool')


def dense_layer(inputs, units, name, activation=None):
    if len(inputs.shape) != 2:
        value = 1
        for i in inputs.shape[1:]:
            value *= i.value
        inputs = tf.reshape(inputs, [-1, value])
    with tf.variable_scope(name):
        input_features = inputs.shape[-1].value
        w = tf.get_variable('weights', [input_features, units],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('biases', [units], initializer=tf.truncated_normal_initializer(stddev=stddev))
        mul = tf.matmul(inputs, w) + b
        if activation is not None:
            mul = activation(mul)
        return mul


def dropout_layer(inputs, keep_prob, training, name):
    with tf.name_scope(name):
        return tf.cond(training, lambda: tf.nn.dropout(inputs, keep_prob, name='dropout'),
                       lambda: inputs)


def cal_accuracy(result_tensor, ground_truth_tensor):
    """Inserts the operations we need to evaluate the accuracy of our results."""
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(result_tensor, 1), tf.argmax(ground_truth_tensor, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass