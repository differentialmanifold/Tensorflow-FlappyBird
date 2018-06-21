import tensorflow as tf
import os
from model import layer_utils
from model import losses


class BaseModel:
    def __init__(self, inputs_shape, n_class, dropout, learning_rate, loss_name):
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.inputs_shape = inputs_shape
        self.n_class = n_class
        self.loss_name = loss_name

        parent_dir = os.path.dirname(os.path.dirname(__file__))
        self.train_writer = tf.summary.FileWriter(os.path.join(parent_dir, 'my_graph', self.name, 'train'))
        self.validation_writer = tf.summary.FileWriter(os.path.join(parent_dir, 'my_graph', self.name, 'validation'))

        self.checkpoint_dir = os.path.join(parent_dir, 'checkpoints', self.name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint')

        self.load()
        self.saver = tf.train.Saver(max_to_keep=1)

    def build(self):
        raise NotImplementedError

    def load(self):
        with tf.name_scope('data'):
            self.X = tf.placeholder(tf.float32, [None] + self.inputs_shape, name="X_placeholder")
            self.Y = tf.placeholder(tf.float32, [None, self.n_class], name="Y_placeholder")

        self.training = tf.placeholder(tf.bool, name='training')
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        self.build()

        with tf.name_scope('loss'):
            self.loss = losses.identify[self.loss_name](self.logits, self.Y)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        self.accuracy = layer_utils.cal_accuracy(self.logits, self.Y)

        self.summaries = tf.summary.merge([
            tf.summary.scalar('loss', self.loss),
            tf.summary.scalar('accuracy', self.accuracy)
        ])

    def predict(self, sess, s):
        return sess.run(self.logits, {self.X: s, self.training: False})

    def update(self, sess, s, y):
        feed_dict = {self.X: s, self.Y: y, self.training: True}
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.train.get_global_step(), self.optimizer, self.loss],
            feed_dict)
        if self.train_writer:
            self.train_writer.add_summary(summaries, global_step)
        return loss

    def restore(self, sess):
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        # if that checkpoint exists, restore from checkpoint
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            print('restore from checkpoints')

    def save(self, sess):
        self.saver.save(sess, self.checkpoint_path, self.global_step)
