#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda
Ad maiorem Dei gloriam
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import pickle
import numpy as np
import tensorflow as tf


BOARD_SIZE = 20*10
PIECE_SIZE = 1
SHIFT_SIZE = 10
ROTATE_SIZE = 4


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

    # Compile model and loss functions
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
        loss=[loss_fn, loss_fn],
        metrics=['accuracy'])

    # Read data and fit model
    x_train, y_train = load_data()
    model.fit(x_train, y_train, epochs=5)


def load_data():
    with open("only_wins.pickle", "rb") as f:
        data = pickle.load(f)

    data_input = []
    data_output = []
    for i, o in data:
        data_input.append(np.array(i))
        data_output.append(np.array(o))

    print("Single input size: %d", len(data_input[0]))
    print("Single output size: %d", len(data_output[1]))

    return np.array(data_input), np.array(data_output)



if __name__ == '__main__':
    main()
