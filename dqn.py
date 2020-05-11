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
        # Input layer
        inputs = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), input_shape=(BOARD_WIDTH, BOARD_HEIGHT))

        # Hidden layers
        hidden_1 = tf.keras.layers.Activation(activation='relu')(inputs)
        hidden_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(hidden_1)
        hidden_3 = tf.keras.layers.Dropout(rate=0.2)(hidden_2)

        hidden_4 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3))(hidden_3)
        hidden_5 = tf.keras.layers.Activation(activation='relu')(hidden_4)
        hidden_6 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(hidden_5)
        hidden_7 = tf.keras.layers.Dropout(rate=0.2)(hidden_6)

        hidden_8 = tf.keras.layers.Flatten()(hidden_7)
        hidden_9 = tf.keras.layers.Dense(units=64)(hidden_8)

        # Output layer
        outputs = tf.keras.layers.Dense(units=ACTION_SPACE_SIZE, activation='linear')(hidden_9)

        # All layers together in model
        model = tf.keras.models.Model(inputs=inputs, outputs=[outputs])

        # Compile model
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer='adam', loss=[loss_fn], metrics=['accuracy'])

        return model


def main():
    pass


if __name__ == '__main__':
    main()
