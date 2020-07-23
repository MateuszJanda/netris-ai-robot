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
import argparse
import pickle


# Netris/environment parameters
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
ACTION_SPACE_SIZE = 4*10
SHFIT_OFFSET = 5
HOST = "127.0.0.1"
PORT = 9800

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


SNAPSHOT = 50
MODEL_SNAPSHOT = "%05d_model.h5"
TARGET_MODEL_SNAPSHOT = "%05d_target_model.h5"
DATA_SNAPSHOT = "%05d_data.pickle"


def main():
    args = parse_args()

    # Fixed memory limit to prevent crash
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        agent = Agent()
        start_episode = 0
        if args.episode:
            load(agent, args.episode)
            start_episode = args.episode + 1

        sock.bind((HOST, args.port))
        sock.listen()
        log("Starting server at %s:%d" % (HOST, args.port))
        env = Environment(sock)

        learn(env, agent, start_episode)

        env.close()


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Netris proxy robot for reinforcement learning (DQN)\n'
                'Mateusz Janda (c) <mateusz janda at gmail com>\n'
                'netris-ai-robot project github.com/MateuszJanda/sloper\n'
                '\n'
                'Robot is waiting for connection from Agent at ' + HOST + ':' + str(PORT) + '\n',
        usage='Please try to use -h, --help for more informations',
        epilog=' \n',
        formatter_class=CustomFormatter)

    parser.add_argument('-e', '--load_episode', required=False, action='store', dest='episode',
                        help='Load data from idicated episode')
    parser.add_argument('-p', '--port', required=False, action='store', default=PORT, dest='port',
                        help='Listen at port')

    args = parser.parse_args()
    args.port = int(args.port)

    if args.episode:
        args.episode = int(args.episode)

    return args


def learn(env, agent, start_episode):
    """Learn though episodes."""
    for episode in range(start_episode, EPISODES):
        episode_reward = play_one_game(env, agent)

        if episode > 0 and episode % SNAPSHOT == 0:
            save(agent, episode)

        log("Episode %d, reward %0.2f, moves %d, avg handling time: %0.4f, game time: %0.4f"
            % (episode,
                episode_reward,
                len(env.handling_time),
                sum(env.handling_time) / len(env.handling_time),
                time.time() - env.game_tic))


def play_one_game(env, agent):
    """Play one game."""
    episode_reward = 0
    epsilon = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # if len(agent.replay_memory) >= MIN_REPLAY_MEMORY_SIZE:
    #     log("Enought data in replay memory. Learning started.")

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

    return episode_reward


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
        self.buffer = bytes()

        self.step_tic = time.time()
        self.game_tic = time.time()
        self.handling_time = []

    def reset(self):
        """
        Reset game. Close connection with old robot and establish
        connection with new one.
        """
        self.game_tic = time.time()
        self.handling_time = []
        self.conn, addr = self.sock.accept()
        done_status, reward, state = self._recevie_data()

        return state

    def step(self, action):
        """Send action to robot and receive new feedback."""
        if action >= ACTION_SPACE_SIZE:
            raise Exception("Action not in action space")

        self.handling_time.append(time.time() - self.step_tic)
        shift = action % BOARD_WIDTH - SHFIT_OFFSET
        rotate = action // BOARD_WIDTH

        message = str(shift) + ' ' + str(rotate) + '\n'
        self.conn.sendall(message.encode())

        done_status, reward, state = self._recevie_data()

        return done_status, reward, state

    def close(self):
        """Close connection with robot."""
        if self.conn:
            self.conn.close()

    def _recevie_data(self):
        """Receive data from robot."""
        if not self.conn:
            raise Exception('Connection not established')

        # Ensure that new full data is received (single line with \n at the end)
        while True:
            self.buffer += self.conn.recv(1024)

            if b'\n' in self.buffer:
                break

        self.step_tic = time.time()

        msg = self.buffer[:self.buffer.find(b'\n')]
        self.buffer = self.buffer[self.buffer.find(b'\n') + 1:]

        # Parse msg from robot
        done_status, reward, *state = msg.decode().split()

        done_status = True if int(done_status) else False
        reward = float(reward)
        state = np.array([float(val) for val in state])

        # if done_status:
        #     log("Game is over")

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
        model.add(tf.keras.layers.Activation(activation='relu'))

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
        state = state.reshape(1, BOARD_HEIGHT, BOARD_WIDTH, 1)
        return self.model.predict(state)[0]

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
        current_states = current_states.reshape(MINIBATCH_SIZE, BOARD_HEIGHT, BOARD_WIDTH, 1)
        current_qs_list = self.model.predict(current_states)

        # Get future states from mini-batch, then query NN model for Q values
        # When using target NN, query it, otherwise main network should be queried
        new_states = np.array([transition.new_state for transition in minibatch])
        new_states = new_states.reshape(MINIBATCH_SIZE, BOARD_HEIGHT, BOARD_WIDTH, 1)
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
            current_qs[transition.action] = new_q

            # Append to training data
            states.append(transition.current_state)
            qs.append(current_qs)

        # Fit on all samples as one batch
        states = np.array(states).reshape(MINIBATCH_SIZE, BOARD_HEIGHT, BOARD_WIDTH, 1)
        self.model.fit(x=states, y=np.array(qs), batch_size=MINIBATCH_SIZE,
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


def save(agent, episode):
    """Save shapshot."""
    agent.model.save_weights(MODEL_SNAPSHOT % episode, save_format="h5")
    agent.target_model.save_weights(TARGET_MODEL_SNAPSHOT % episode, save_format="h5")

    with open(DATA_SNAPSHOT % episode, "wb") as f:
        pickle.dump((agent.target_update_counter, agent.replay_memory), f)


def load(agent, episode):
    """Load shapshot."""
    agent.model.load_weights(MODEL_SNAPSHOT % episode)
    agent.target_model.load_weights(TARGET_MODEL_SNAPSHOT % episode)

    with open(DATA_SNAPSHOT % episode, "rb") as f:
        agent.target_update_counter, agent.replay_memory = pickle.load(f)


def print_board(board):
    """Print board state. For debug only."""
    log("Board")
    for line in board:
        l = "".join(["1" if b else " " for b in line])
        log(l)


def log(*args, **kwargs):
    """Print log to other terminal or file."""
    print(*args, **kwargs)


if __name__ == '__main__':
    main()
