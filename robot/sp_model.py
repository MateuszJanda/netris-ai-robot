#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""

import tensorflow as tf
from dqn import config


class SpModel:
    def __init__(self, episode=None):
        """
        Build NN model.
        """
        if episode:
            self._model = tf.keras.models.load_model(config.MODEL_SNAPSHOT % episode)
        else:
            self._model = self.create_model()

        print(self._model.summary())

    @staticmethod
    def create_model():
        """Create tensorflow model."""
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
            input_shape=(config.BOARD_HEIGHT, config.BOARD_WIDTH, 1), activation='relu'))

        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))

        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

        model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        model.add(tf.keras.layers.Dropout(rate=0.25))

        model.add(tf.keras.layers.Dense(units=512, activation='relu'))

        model.add(tf.keras.layers.Dense(units=config.ACTION_SPACE_SIZE, activation='linear'))

        # Compile model
        model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001), loss='mse',
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
        """
        Getter to tensorflow model.
        """
        return self._model
