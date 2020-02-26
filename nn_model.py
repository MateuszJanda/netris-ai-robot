#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda
Ad maiorem Dei gloriam
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import pickle
import random
import numpy as np
import tensorflow as tf


BOARD_SIZE = 20*10
PIECE_SIZE = 1
SHIFT_SIZE = 10
ROTATE_SIZE = 4

PIECE_TYPES = 6


def main():
    # Input layer
    inputs = tf.keras.Input(shape=(BOARD_SIZE + PIECE_SIZE,))

    # Hidden layers
    hidden_1 = tf.keras.layers.Dense(128, activation='relu')(inputs)
    hidden_2 = tf.keras.layers.Dropout(0.2)(hidden_1)

    # Last layer - two separate outputs
    outputs_1 = tf.keras.layers.Dense(SHIFT_SIZE, activation='relu')(hidden_2)
    outputs_2 = tf.keras.layers.Dense(ROTATE_SIZE, activation='relu')(hidden_2)

    # All layers together in model
    model = tf.keras.models.Model(inputs=inputs, outputs=[outputs_1, outputs_2])

    # Compile model
    model.compile(optimizer='adam',
        loss=['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy'],
        metrics=['accuracy'])

    # Read data
    x_train, y_shift_train, y_rotate_train, \
        x_test, y_shift_test, y_rotate_test = load_data()

    # Fit model
    model.fit(x=x_train, y=[y_shift_train, y_rotate_train],
        epochs=45)

    # Evaluate model with test data
    print("\nEvaluate:")
    model.evaluate(x=x_test, y=[y_shift_test, y_rotate_test],
        verbose=2)

    # Summary and save entire model
    print("\nSummary:")
    model.summary()
    model.save("only_wins.h5")


def load_data(split=0.7):
    with open("only_wins.pickle", "rb") as f:
        data = pickle.load(f)

    x_train = []
    y_shift_train = []
    y_rotate_train = []
    x_test = []
    y_shift_test = []
    y_rotate_test = []
    for i, s, r in data:
        # Normalize piece type - float in range [0, 1)
        i[0] = i[0] / PIECE_TYPES

        # Split data between train and test
        if random.random() < split:
            x_train.append(np.array(i))
            y_shift_train.append(s)
            y_rotate_train.append(r)
        else:
            x_test.append(np.array(i))
            y_shift_test.append(s)
            y_rotate_test.append(r)

    x_train = np.array(x_train)
    y_shift_train = np.array(y_shift_train)
    y_rotate_train = np.array(y_rotate_train)
    x_test = np.array(x_test)
    y_shift_test = np.array(y_shift_test)
    y_rotate_test = np.array(y_rotate_test)

    print("\nData info")
    print("x_train shape:", x_train.shape)
    print("y_shift_train shape:", y_shift_train.shape)
    print("y_rotate_train shape:", y_rotate_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_shift_test shape:", y_shift_test.shape)
    print("y_rotate_test shape:", y_rotate_test.shape, "\n")

    return x_train, y_shift_train, y_rotate_train, \
        x_test, y_shift_test, y_rotate_test


if __name__ == '__main__':
    main()