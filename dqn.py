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
BOARD_HEIGHT = 20
BOARD_WIDTH = 10
ACTION_SPACE_SIZE = 4*10
SHFIT_OFFSET = 5
HOST = "127.0.0.1"
PORT = 9800

# DQN parameters
DISCOUNT = 0.95                 # Gamma (𝛾) parameter from Bellman equation
REPLAY_MEMORY_SIZE = 100_000    # Last steps kept for model training
MIN_REPLAY_MEMORY_SIZE = 2_000  # Minimum number of steps in a memory to start training

MINIBATCH_SIZE = 64             # How many steps (samples) to use for training

EPISODES = 20_000               # Episodes == full games

# Exploration settings
EPSILON_DECAY = 0.999           # Try/explore other actions to escape local minimum
MIN_EPSILON = 0.001

# Snapshot settings
SNAPSHOT_MOD = 50
MODEL_SNAPSHOT = "%05d_model.h5"
DATA_SNAPSHOT = "%05d_data.pickle"
STATS_FILE = "stats.txt"


def main():
    args = parse_args()

    # Fixed memory limit to prevent crash
    if args.gpu:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        flat_model = FlatModel()
        agent = Agent(flat_model)
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
    parser.add_argument('-g', '--gpu', required=False, action='store_true', dest='gpu',
                        help='Use GPU (with fixed memory limit to prevent crashes).')
    parser.add_argument('-p', '--port', required=False, action='store', default=PORT, dest='port',
                        help='Listen at port')

    args = parser.parse_args()
    args.port = int(args.port)

    if args.episode:
        args.episode = int(args.episode)

    return args


def learn(env, agent, start_episode):
    """Learn through episodes."""
    for episode in range(start_episode, EPISODES + 1):
        episode_reward = play_one_game(env, agent)

        if episode > 0 and episode % SNAPSHOT_MOD == 0:
            save(agent, episode, episode_reward, len(env.handling_time))

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
    current_state = agent.reshape_input(current_state)

    # if len(agent.replay_memory) >= MIN_REPLAY_MEMORY_SIZE:
    #     log("Enough data in replay memory. Learning started.")

    # Reset flag and start iterating until episode ends
    last_round = False

    while not last_round:
        tic = time.time()
        # Explore other actions with probability epsilon
        if np.random.random() <= epsilon:
            action = np.random.randint(0, ACTION_SPACE_SIZE)
        else:
            action = np.argmax(agent.get_q_values(current_state))

        last_round, reward, next_state = env.step(action)
        next_state = agent.reshape_input(next_state)

        # Transform new continuous state to new discrete state and count reward
        episode_reward += reward

        # Every step update replay memory and train NN model
        transition = Transition(current_state, action, reward, next_state, last_round)
        agent.update_replay_memory(transition)
        agent.train(last_round)

        current_state = next_state

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
        self._sock = sock
        self._conn = None
        self._buffer = bytes()

        self._step_tic = time.time()
        self.game_tic = time.time()
        self.handling_time = []

    def reset(self):
        """
        Reset game. Close connection with old robot and establish
        connection with new one.
        """
        self.game_tic = time.time()
        self.handling_time = []
        self._conn, addr = self._sock.accept()
        last_round, reward, state = self._recevie_data()

        return state

    def step(self, action):
        """Send action to robot and receive new feedback."""
        if action >= ACTION_SPACE_SIZE:
            raise Exception("Action not in action space:", action)

        self.handling_time.append(time.time() - self._step_tic)
        shift = action % BOARD_WIDTH - SHFIT_OFFSET
        rotate = action // BOARD_WIDTH

        message = str(shift) + ' ' + str(rotate) + '\n'
        self._conn.sendall(message.encode())

        last_round, reward, state = self._recevie_data()

        return last_round, reward, state

    def close(self):
        """Close connection with robot."""
        if self._conn:
            self._conn.close()

    def _recevie_data(self):
        """Receive data from robot."""
        if not self._conn:
            raise Exception('Connection not established')

        # Ensure that new full data is received (single line with \n at the end)
        while True:
            self._buffer += self._conn.recv(1024)

            if b'\n' in self._buffer:
                break

        self._step_tic = time.time()

        msg = self._buffer[:self._buffer.find(b'\n')]
        self._buffer = self._buffer[self._buffer.find(b'\n') + 1:]

        # Parse msg from robot
        last_round, reward, *state = msg.decode().split()

        last_round = True if int(last_round) else False
        reward = float(reward)
        state = np.array([float(val) for val in state])

        return last_round, reward, state


class FlatModel:
    def __init__(self):
        # Build NN model
        self._model = self.create_model(BOARD_HEIGHT * BOARD_WIDTH)
        log(self._model.summary())

    @staticmethod
    def create_model(size):
        """Create tensorflow model."""
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Input(shape=(size,)))
        model.add(tf.keras.layers.Dense(units=256, activation='relu'))

        model.add(tf.keras.layers.Dense(units=128, activation='relu'))

        model.add(tf.keras.layers.Dense(units=128, activation='relu'))

        model.add(tf.keras.layers.Dense(units=64, activation='relu'))

        model.add(tf.keras.layers.Dense(units=ACTION_SPACE_SIZE, activation='linear'))

        # Compile model
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.05), loss='mse',
            metrics=['accuracy'])

        return model

    def predict(self, batch_size, state):
        """
        Queries NN model for Q values given current observation (state).
        """
        state = state.reshape(batch_size, BOARD_HEIGHT * BOARD_WIDTH)
        return self._model.predict(state)

    def fit(self, x, y, batch_size, verbose, shuffle):
        """Wrapper around fit."""
        x = np.array(x).reshape(batch_size, BOARD_HEIGHT * BOARD_WIDTH)
        self._model.fit(x=x, y=y, batch_size=batch_size, verbose=verbose,
            shuffle=shuffle)

    def get_tf_model(self):
        """Getter to tensorflow model."""
        return self._model

    def reshape_input(self, state):
        """Just return same state."""
        return state


class CnnModel:
    def __init__(self):
        # Board size with extra padding
        self._height = BOARD_HEIGHT + 2
        self._width = BOARD_WIDTH + 2

        # Build NN model
        self._model = self.create_model(self._height, self._width)

    @staticmethod
    def create_model(height, width):
        """Create tensorflow model."""
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
        model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),
            input_shape=(height, width, 1), activation='relu'))

        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))

        model.add(tf.keras.layers.Dense(units=ACTION_SPACE_SIZE, activation='linear'))

        # Compile model
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.05), loss='mse',
            metrics=['accuracy'])

        return model

    def predict(self, batch_size, state):
        """
        Queries NN model for Q values given current observation (state).
        """
        state = state.reshape(batch_size, self._height, self._width, 1)
        return self._model.predict(state)

    def fit(self, x, y, batch_size, verbose, shuffle):
        """Wrapper around fit."""
        x = np.array(x).reshape(batch_size, self._height, self._width, 1)
        self._model.fit(x=x, y=y, batch_size=batch_size, verbose=verbose,
            shuffle=shuffle)

    def get_tf_model(self):
        """Getter to tensorflow model."""
        return self._model

    def reshape_input(self, state):
        """
        Board state with extra padding, because CNN remove boarded where
        piece data are stored.
        """
        state = sate.reshape(BOARD_HEIGHT, BOARD_WIDTH)
        return np.pad(state, pad_width=1, mode='constant', constant_values=0)


class Agent:
    """DQN agent."""

    def __init__(self, model):
        # Build NN model
        self._model = model

        # An array with last REPLAY_MEMORY_SIZE steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

    def update_replay_memory(self, transition):
        """Adds transition (step's data) to a replay memory."""
        self.replay_memory.append(transition)

    def get_q_values(self, state):
        """
        Queries NN model for Q values given current observation (state).
        """
        return self._model.predict(batch_size=1, state=state)[0]

    def train(self, last_round):
        """Trains NN model every step during episode."""

        # Start training only if certain number of samples is already saved in
        # replay memory
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a mini-batch of random samples from replay memory
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_q_values, future_q_values = self.query_model_for_q_values(minibatch)

        states = []    # Input X
        q_values = []  # Output y

        for index, transition in enumerate(minibatch):
            # If last round assign reward to Q
            if transition.last_round:
                new_q = transition.reward
            # Otherwise set new Q from future states (Bellman equation)
            else:
                max_future_q = np.max(future_q_values[index])
                new_q = transition.reward + DISCOUNT * max_future_q

            # Update Q value for given action, and append to training output (y) data
            current_qs = current_q_values[index]
            current_qs[transition.action] = new_q
            q_values.append(current_qs)

            # Append to training input (X) data
            states.append(transition.current_state)

        # Fit with new Q values
        self._model.fit(x=states, y=np.array(q_values), batch_size=MINIBATCH_SIZE,
            verbose=0, shuffle=False)

    def query_model_for_q_values(self, minibatch):
        """
        Take current and next states (from minibach) and query NN model for Q
        values.
        """
        current_states = np.array([transition.current_state for transition in minibatch])
        current_q_values = self._model.predict(MINIBATCH_SIZE, current_states)

        next_states = np.array([transition.next_state for transition in minibatch])
        future_q_values = self._model.predict(MINIBATCH_SIZE, next_states)

        return current_q_values, future_q_values

    def get_tf_model(path):
        """Getter to tensorflow model."""
        return self._model.get_tf_model()

    def reshape_input(self, state):
        """Reshape input state if needed later by model."""
        return self._model.reshape_input(state)


class Transition:
    def __init__(self, current_state, action, reward, next_state, last_round):
        self.current_state = current_state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.last_round = last_round


def save(agent, episode, episode_reward, moves):
    """Save snapshot."""
    agent.get_tf_model().save_weights(MODEL_SNAPSHOT % episode, save_format="h5")

    with open(DATA_SNAPSHOT % episode, "wb") as f:
        pickle.dump((agent.replay_memory, episode_reward), f)

    with open(STATS_FILE, "a") as f:
        f.write("Episode: %d, moves: %d, reward: %0.2f\n" % (episode, moves, episode_reward))


def load(agent, episode):
    """Load snapshot."""
    agent.get_model().load_weights(MODEL_SNAPSHOT % episode)

    with open(DATA_SNAPSHOT % episode, "rb") as f:
        agent.replay_memory, _ = pickle.load(f)


def print_board(board, height, width):
    """Print board state. For debug only."""
    log("Board")
    board = board.reshape(height, width)
    for line in board:
        l = "".join(["1" if b else " " for b in line])
        log("|" + l + "|")


def log(*args, **kwargs):
    """Print log to other terminal or file."""
    print(*args, **kwargs)


if __name__ == '__main__':
    main()
