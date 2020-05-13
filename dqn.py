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

REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000


class Agent:
    """DQN agent."""
    def __init__(self):
        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
         model = tf.keras.models.Sequential()

        # https://missinglink.ai/guides/tensorflow/tensorflow-conv2d-layers-practical-guide/
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D#arguments
        #
        # input_shape: tensor input become 4D:
        # [batch_size, input_height, input_width, input_channels]
        #
        # filters: (integer) the dimensionality of the output space. Here for
        # each pixel there will be generated 256 features.
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

    def update_replay_memory(self, transition):
        """Adds step's data to a memory replay array."""
        self.replay_memory.append(transition)

    def get_qs(self, state):
        """Queries main model for Q values given current observation (state)."""
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]


class Transition:
    def __init__(self, state, action, reward, new_state, done_status):
        self.state = state
        self.action = action
        self.reward = reward
        self.new_state = new_state
        self.done_status = done_status


def main():
    pass


if __name__ == '__main__':
    main()
