#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda
Ad maiorem Dei gloriam
"""


"""
Reinforcement learning - Deep Q-Network/Learning.

Useful links:
- "Playing Atari with Deep Reinforcement Learning" by
    Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves,
    Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller; DeepMind Technologies
    https://arxiv.org/pdf/1312.5602.pdf
- "Training Deep Q Learning and Deep Q Networks (DQN) Intro and Agent -
    Reinforcement Learning w/ Python Tutorial" by Harrison Kinsley
    https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/
- https://www.quora.com/Artificial-Intelligence-What-is-an-intuitive-explanation-of-how-deep-Q-networks-DQN-work
- "How to use Q Learning in Video Games Easily" by Siraj Raval
    https://www.youtube.com/watch?v=A5eihauRQvo
"""


import numpy as np
import tensorflow as tf

BOARD_WIDTH = 10
BOARD_HEIGHT = 20

ACTION_SPACE_SIZE = 4*10


class Agent:
    """DQN agent."""

    def create_model(self):
         model = tf.keras.models.Sequential()

        # https://missinglink.ai/guides/tensorflow/tensorflow-conv2d-layers-practical-guide/
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D#arguments
        #
        # input_shape: tensor input become 4D: [batch_size, in_height, in_width, in_channels]
        # filters: Integer, the dimensionality of the output space
        #   for each pixel there will be generated 256 features.
        model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), input_shape=(1, BOARD_HEIGHT, BOARD_WIDTH, 1)))
        model.add(tf.keras.layers.Activation(activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(rate=0.2))

        model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3)))
        model.add(tf.keras.layers.Activation(activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(rate=0.2))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=64))

        model.add(tf.keras.layers.Dense(units=ACTION_SPACE_SIZE, activation='linear'))

        # Compile model
        model.compile(optimizer=Adam(lr=0.001), loss='mse', metrics=['accuracy'])

        return model


def main():
    pass


if __name__ == '__main__':
    main()
