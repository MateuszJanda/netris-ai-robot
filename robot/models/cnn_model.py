#!/usr/bin/env python3

# Author: Mateusz Janda <mateusz janda at gmail com>
# Site: github.com/MateuszJanda/netris-ai-robot
# Ad maiorem Dei gloriam

import tensorflow as tf
from robot import config


class CnnModel:
    """
    Model with 2D convolution layers.
    """

    def __init__(self, episode=None):
        if episode:
            self._model = tf.keras.models.load_model(config.MODEL_SNAPSHOT % episode)
        else:
            self._model = self.create_model()

        print(self._model.summary())

    @staticmethod
    def create_model():
        """Create tensorflow model."""
        model = tf.keras.models.Sequential()

        # Conv2D:
        # - https://towardsdatascience.com/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148
        # - https://missinglink.ai/guides/tensorflow/tensorflow-conv2d-layers-practical-guide/
        # - https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D#arguments
        #
        # input_shape: tensor input become 4D: [batch_size, input_height, input_width, input_channels]
        #   when layer is added to model batch_size is not needed.
        #
        # filters: (integer) the dimensionality of the output space. Here for
        #   each pixel there will be generated 256 features.
        model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same',
            input_shape=(config.BOARD_HEIGHT, config.BOARD_WIDTH, 1), activation='relu'))

        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))

        model.add(tf.keras.layers.Dense(units=config.ACTION_SPACE_SIZE, activation='linear'))

        # Compile model
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mse',
            metrics=['accuracy'])

        return model

    def predict(self, batch_size, state):
        """
        Queries NN model for Q values given current observation (state).
        """
        state = state.reshape(batch_size, config.BOARD_HEIGHT, config.BOARD_WIDTH, 1)
        return self._model.predict(state)

    def fit(self, x, y, batch_size, verbose, shuffle):
        """Wrapper around fit."""
        x = x.reshape(batch_size, config.BOARD_HEIGHT, config.BOARD_WIDTH, 1)
        self._model.fit(x=x, y=y, batch_size=batch_size, verbose=verbose,
            shuffle=shuffle)

    def get_tf_model(self):
        """Getter to tensorflow model."""
        return self._model
