import tensorflow as tf

from model import layer_utils

from model.base_model import BaseModel


class Simple(BaseModel):
    def __init__(self, inputs_shape, n_class, dropout=0.75, learning_rate=0.001,
                 loss_name='mean_softmax_cross_entropy'):
        self.name = 'simple'
        super().__init__(inputs_shape, n_class, dropout, learning_rate, loss_name)

    def build(self):
        self.conv1 = layer_utils.conv_layer(self.X, out_channels=32, kernel_size=[8, 8], name='conv1', stride=4)

        self.conv2 = layer_utils.conv_layer(self.conv1, out_channels=64, kernel_size=[4, 4], name='conv2', stride=2)

        self.conv3 = layer_utils.conv_layer(self.conv2, out_channels=64, kernel_size=[3, 3], name='conv3', stride=1)

        self.fc = layer_utils.dense_layer(self.conv3, units=512, name='fc', activation=tf.nn.relu)

        self.logits = layer_utils.dense_layer(self.fc, units=self.n_class, name='softmax_linear')
