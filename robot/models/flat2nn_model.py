#!/usr/bin/env python3

# Author: Mateusz Janda <mateusz janda at gmail com>
# Site: github.com/MateuszJanda/netris-ai-robot
# Ad maiorem Dei gloriam

import tensorflow as tf
from robot import config


class Flat2NnModel:
    """
    Flat model with with more layers than FlatNN.
    """

    def __init__(self, episode=None, learning_rate=0.001):
        self._learning_rate = learning_rate

        # Build NN model
        if episode:
            self._model = tf.keras.models.load_model(config.MODEL_SNAPSHOT % episode)
        else:
            self._model = self.create_model()

        print(self._model.summary())

    def create_model(self):
        """Create tensorflow model."""
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Input(shape=(config.BOARD_HEIGHT * config.BOARD_WIDTH,)))
        model.add(tf.keras.layers.Dense(units=256, activation='relu'))

        model.add(tf.keras.layers.Dense(units=128, activation='relu'))

        model.add(tf.keras.layers.Dense(units=128, activation='relu'))

        model.add(tf.keras.layers.Dense(units=128, activation='relu'))

        model.add(tf.keras.layers.Dense(units=128, activation='relu'))

        model.add(tf.keras.layers.Dense(units=128, activation='relu'))

        model.add(tf.keras.layers.Dense(units=128, activation='relu'))

        model.add(tf.keras.layers.Dense(units=64, activation='relu'))

        model.add(tf.keras.layers.Dense(units=config.ACTION_SPACE_SIZE, activation='linear'))

        # Compile model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self._learning_rate),
            loss='mse', metrics=['accuracy'])

        return model

    def predict(self, batch_size, state):
        """
        Queries NN model for Q values given current observation (state).
        """
        state = state.reshape(batch_size, config.BOARD_HEIGHT * config.BOARD_WIDTH)
        return self._model.predict(state)

    def fit(self, x, y, batch_size, verbose, shuffle):
        """Wrapper around fit."""
        x = x.reshape(batch_size, config.BOARD_HEIGHT * config.BOARD_WIDTH)
        self._model.fit(x=x, y=y, batch_size=batch_size, verbose=verbose,
            shuffle=shuffle)

    def get_tf_model(self):
        """
        Getter to tensorflow model.
        """
        return self._model
