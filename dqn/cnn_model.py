#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""

import tensorflow as tf
import numpy as np
import config
import utils


class CnnModel:
    def __init__(self, episode=None):
        # Board size with extra padding
        self._height = config.BOARD_HEIGHT + 2
        self._width = config.BOARD_WIDTH + 2

        if episode:
            self._model = tf.keras.models.load_model(config.MODEL_SNAPSHOT % episode)
        else:
            self._model = self.create_model(self._height, self._width)

        utils.log(self._model.summary())

    @staticmethod
    def create_model(height, width):
        """Create tensorflow model."""
        model = tf.keras.models.Sequential()

        # Conv2D:
        # - https://missinglink.ai/guides/tensorflow/tensorflow-conv2d-layers-practical-guide/
        # - https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D#arguments
        #
        # input_shape: tensor input become 4D: [batch_size, input_height, input_width, input_channels]
        #   when layer is added to model batch_size is not needed.
        #
        # filters: (integer) the dimensionality of the output space. Here for
        #   each pixel there will be generated 256 features.
        model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),
            input_shape=(height, width, 1), activation='relu'))

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
        state = state.reshape(batch_size, self._height, self._width, 1)
        return self._model.predict(state)

    def fit(self, x, y, batch_size, verbose, shuffle):
        """Wrapper around fit."""
        x = x.reshape(batch_size, self._height, self._width, 1)
        self._model.fit(x=x, y=y, batch_size=batch_size, verbose=verbose,
            shuffle=shuffle)

    def get_tf_model(self):
        """Getter to tensorflow model."""
        return self._model

    def reshape_input(self, state):
        """
        Board state with extra padding, because CNN remove boarded where
        piece data are stored.
        """
        state = state.reshape(config.BOARD_HEIGHT, config.BOARD_WIDTH)
        return np.pad(state, pad_width=1, mode='constant', constant_values=0)
