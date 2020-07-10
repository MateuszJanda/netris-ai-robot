#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda
Ad maiorem Dei gloriam
"""


"""
Reinforcement learning - Deep Q-Network/Learning.

Credits:
- "Playing Atari with Deep Reinforcement Learning" by
    Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves,
    Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller; DeepMind Technologies
    https://arxiv.org/pdf/1312.5602.pdf
- "Human-level control through deep reinforcement learning" by
    Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness,
    Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland,
    Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou,
    Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg & Demis Hassabis
    https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
- "Training Deep Q Learning and Deep Q Networks (DQN) Intro and Agent -
    Reinforcement Learning w/ Python Tutorial" by Harrison Kinsley
    https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/
- https://www.quora.com/Artificial-Intelligence-What-is-an-intuitive-explanation-of-how-deep-Q-networks-DQN-work
- "How to use Q Learning in Video Games Easily" by Siraj Raval
    https://www.youtube.com/watch?v=A5eihauRQvo
- Q-learning
    https://en.wikipedia.org/wiki/Q-learning#Algorithm
- Bellman equation
    https://en.wikipedia.org/wiki/Bellman_equation
"""

import tensorflow as tf
import numpy as np
import random
from collections import deque
import socket
import time


# Netris/environment parameters
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
ACTION_SPACE_SIZE = 4*10
SHFIT_OFFSET = 5
HOST = "127.0.0.1"
PORT = 9898

# DQN parameters
DISCOUNT = 0.99                 # Gamma (𝛾) parameter from Bellman equation
REPLAY_MEMORY_SIZE = 50_000     # Last steps kept for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training

MINIBATCH_SIZE = 64             # How many steps (samples) to use for training
UPDATE_TARGET = 5               # Copy weights every UPDATE_TARGET finished games

EPISODES = 20_000               # Episodes == full games

# Exploration settings
EPSILON_DECAY = 0.999           # Try/explore other actions to escape local minimum
MIN_EPSILON = 0.001


def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, PORT))
        sock.listen()

        env = Environment(sock)
        agent = Agent()

        learn(env, agent)

        env.close()


def learn(env, agent):
    """Learn though episodes."""
    for _ in range(EPISODES):
        play_one_game(env, agent)


def play_one_game(env, agent):
    """Play one game."""
    # Reset episode reward
    episode_reward = 0
    epsilon = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done_status = False

    while not done_status:
        tic = time.time()
        # Explore other actions with probability 1 - epsilon
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, ACTION_SPACE_SIZE)

        done_status, reward, new_state = env.step(action)

        # Transform new continuous state to new discrete state and count reward
        episode_reward += reward

        # Every step update replay memory and train main NN model
        transition = Transition(current_state, action, reward, new_state, done_status)
        agent.update_replay_memory(transition)
        agent.train(done_status)

        current_state = new_state

        epsilon = adjust_epsilon(epsilon)

        log("One round", time.time() - tic)


def adjust_epsilon(epsilon):
    """Decay epsilon."""
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

    return epsilon


class Environment:
    def __init__(self, sock):
        self.sock = sock
        self.conn = None

    def reset(self):
        """Reset game."""
        print("Reset game")
        self.conn, addr = self.sock.accept()
        done_status, reward, state = self._recevie_data()

        return state

    def step(self, action):
        tic = time.time()
        if action >= ACTION_SPACE_SIZE:
            raise Exception("Action not in action space")

        shift = action % BOARD_WIDTH - SHFIT_OFFSET
        rotate = action // BOARD_WIDTH

        message = str(shift) + ' ' + str(rotate) + '\n'
        self.conn.sendall(message.encode())

        done_status, reward, state = self._recevie_data()

        delay = time.time() - tic
        print("Step delay", delay)
        return done_status, reward, state

    def close(self):
        if self.conn:
            self.conn.close()

    def _recevie_data(self):
        if not self.conn:
            raise Exception('Connection not established')

        data = bytes()
        while True:
            data += self.conn.recv(1024)

            if b'\n' in data:
                break

        # log('data decode', data.decode())
        done_status, reward, *state = data.decode().split()

        done_status = True if int(done_status) else False
        reward = int(reward)
        state = np.array([float(val) for val in state])
        # log(done_status, reward, state)

        return done_status, reward, state


class Agent:
    """DQN agent."""

    def __init__(self):
        # Build main NN model
        self.model = self.create_model()

        # Build target NN model
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last REPLAY_MEMORY_SIZE steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target NN with main NN weights
        self.target_update_counter = 0

    def create_model(self):
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
        model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3),
            input_shape=(BOARD_HEIGHT, BOARD_WIDTH, 1)))
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
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mse',
            metrics=['accuracy'])

        return model

    def update_replay_memory(self, transition):
        """Adds transition (step's data) to a replay memory."""
        self.replay_memory.append(transition)

    def get_qs(self, state):
        """
        Queries main NN model for Q values given current observation (state).
        Also flatten output - from (1, ACTION_SPACE_SIZE) shape to
        (ACTION_SPACE_SIZE,)
        """
        aaa = np.array(state)
        print("Array shape:", state.shape, aaa.shape)
        return self.model.predict(aaa)[0]

    def train(self, done_status):
        """Trains main NN model every step during episode."""

        # Start training only if certain number of samples is already saved in
        # replay memory
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a mini-batch of random samples from replay memory
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from mini-batch, then query NN model for Q values
        current_states = np.array([transition.current_state for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from mini-batch, then query NN model for Q values
        # When using target NN, query it, otherwise main network should be queried
        new_states = np.array([transition.new_state for transition in minibatch])
        future_qs_list = self.target_model.predict(new_states)

        # Input (x), and output (y) for training
        states = []
        qs = []

        for index, transition in enumerate(minibatch):
            # If not a terminal state then get new Q from future states (Bellman equation)
            if not transition.done_status:
                max_future_q = np.max(future_qs_list[index])
                new_q = transition.reward + DISCOUNT * max_future_q
            # Otherwise set to new Q reward
            else:
                new_q = transition.reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # Append to training data
            states.append(current_state)
            qs.append(current_qs)

        # Fit on all samples as one batch
        self.model.fit(x=np.array(states), y=np.array(qs), batch_size=MINIBATCH_SIZE,
            verbose=0, shuffle=False)

        self.update_weights(done_status)


    def update_weights(self, done_status):
        """Update weights when counter reaches certain value."""
        # Update target NN counter every episode
        if done_status:
            self.target_update_counter += 1

        # Update weights every UPDATE_TARGET end games
        if self.target_update_counter > UPDATE_TARGET:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


class Transition:
    def __init__(self, current_state, action, reward, new_state, done_status):
        self.current_state = current_state
        self.action = action
        self.reward = reward
        self.new_state = new_state
        self.done_status = done_status


def log(*args, **kwargs):
    """Print log to other terminal or file."""
    print(*args, **kwargs)


if __name__ == '__main__':
    main()
