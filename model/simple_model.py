import tensorflow as tf

from model import layer_utils

from model.base_model import BaseModel

# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation, Flatten
# from keras.layers.convolutional import Convolution2D, MaxPooling2D
# from keras.optimizers import SGD, Adam
# import os

# img_rows, img_cols = 80, 80
# # Convert image into Black and white
# img_channels = 4  # We stack 4 frames
# LEARNING_RATE = 1e-4


# def buildmodel():
#     print("Now we build the model")
#     model = Sequential()
#     model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',
#                             input_shape=(img_rows, img_cols, img_channels)))  # 80*80*4
#     model.add(Activation('relu'))
#     model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
#     model.add(Activation('relu'))
#     model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
#     model.add(Activation('relu'))
#     model.add(Flatten())
#     model.add(Dense(512))
#     model.add(Activation('relu'))
#     model.add(Dense(2))
#
#     adam = Adam(lr=LEARNING_RATE)
#     model.compile(loss='mse', optimizer=adam)
#     print("We finish building the model")
#     return model


class Simple(BaseModel):
    def __init__(self, inputs_shape, n_class, dropout=0.75, learning_rate=0.001,
                 loss_name='mean_softmax_cross_entropy'):
        self.name = 'simple'
        super().__init__(inputs_shape, n_class, dropout, learning_rate, loss_name)
        # self.build_from_model()

    def build(self):
        self.conv1 = layer_utils.conv_layer(self.X, out_channels=32, kernel_size=[8, 8], name='conv1', stride=4)

        self.conv2 = layer_utils.conv_layer(self.conv1, out_channels=64, kernel_size=[4, 4], name='conv2', stride=2)

        self.conv3 = layer_utils.conv_layer(self.conv2, out_channels=64, kernel_size=[3, 3], name='conv3', stride=1)

        self.fc = layer_utils.dense_layer(self.conv3, units=512, name='fc', activation=tf.nn.relu)

        self.logits = layer_utils.dense_layer(self.fc, units=self.n_class, name='softmax_linear')

        # def build_from_model(self):
        #
        #     e1_params = [t for t in tf.trainable_variables()]
        #     model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model.h5')
        #     model = buildmodel()
        #     model.load_weights(model_path)
        #
        #     my_weights = model.get_weights()
        #
        #     for i in range(len(e1_params)):
        #         print(e1_params[i])
        #         print(my_weights[i].shape)
        #         print('build sucessfully')
        #
        #     self.update_ops = []
        #     for i in range(len(e1_params)):
        #         op = tf.assign(e1_params[i], my_weights[i])
        #         self.update_ops.append(op)
