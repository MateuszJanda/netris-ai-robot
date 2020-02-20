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
    x_train, y_train = load_data()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(BOARD_SIZE + PIECE_SIZE)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(SHIFT_SIZE + ROTATE_SIZE)
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer='adam',
        loss=loss_fn,
        metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)

    model.evaluate(x_test,  y_test, verbose=2)


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
