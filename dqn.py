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

from collections import deque
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
        """Adds transition (step's data) to a replay memory."""
        self.replay_memory.append(transition)

    def get_qs(self, state):
        """
        Queries main NN model for Q values given current observation (state)."""
        return self.model.predict(np.array(state))[0]

    def train(self, terminal_state):
        """Trains main NN model every step during episode."""

        # Start training only if certain number of samples is already saved in
        # replay memory
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from replay memory
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition.current_state for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target NN, query it, otherwise main network should be queried
        new_states = np.array([transition.new_state for transition in minibatch])
        future_qs_list = self.target_model.predict(new_states)

        X = []
        y = []

        for index, transition in enumerate(minibatch):
            # If not a terminal state, get new q from future states,
            # otherwise set it to reward
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = transition.reward + DISCOUNT * max_future_q
            else:
                new_q = transition.reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # Append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target NN with weights of
        # main NN
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


class Transition:
    def __init__(self, current_state, action, reward, new_state, done_status):
        self.current_state = current_state
        self.action = action
        self.reward = reward
        self.new_state = new_state
        self.done_status = done_status


def main():
    pass


if __name__ == '__main__':
    main()
