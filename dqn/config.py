#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""

import tensorflow as tf
import argparse
import socket
import time
import pickle
from dqn.envirement import Environment


HOST = "127.0.0.1"
PORT = 9800

# Netris/environment parameters
BOARD_HEIGHT = 20
BOARD_WIDTH = 10
ACTION_SPACE_SIZE = 4 * 10
SHFIT_OFFSET = 5

SCREEN_ID = 0
TOP_LINE = 19
BOTTOM_LINE = 0
EMPTY_BLOCK = 0
FULL_BLOCK = 1
EMPTY_LINE = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


SHFIT_OFFSET = 5
MAX_PIECE_HEIGHT = 4
MAX_ALLOWED_GAPS = 5
MAX_CLEARED_LINES = 4
REWAD_THRESHOLDS = MAX_CLEARED_LINES + (MAX_CLEARED_LINES - 1)

# Snapshot settings
SNAPSHOT_MODULO = 100
MODEL_SNAPSHOT = "%05d_model.h5"
DATA_SNAPSHOT = "%05d_data.pickle"
STATS_FILE = "stats.txt"

# DQN parameters
DISCOUNT = 0.95                               # Gamma (𝛾) parameter from Bellman equation
MINIBATCH_SIZE = 128                          # How many steps (samples) to use for training
REPLAY_MEMORY_SIZE = 40_000                   # Last steps kept for model training
MIN_REPLAY_MEMORY_SIZE = 10 * MINIBATCH_SIZE  # Minimum number of steps in a memory to start training
EPISODES = 50_000                             # Episodes == full games


class Transition:
    def __init__(self, current_state, action, reward, next_state, last_round):
        self.current_state = current_state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.last_round = last_round


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Netris proxy robot for reinforcement learning (DQN)\n'
                'Mateusz Janda (c) <mateusz janda at gmail com>\n'
                'netris-ai-robot project github.com/MateuszJanda/netris-ai-robot\n'
                '\n'
                'Robot is waiting for connection from Agent at ' + HOST + ':' + str(PORT) + '\n',
        usage='Please try to use -h, --help for more informations',
        epilog=' \n',
        formatter_class=CustomFormatter)

    parser.add_argument('-e', '--load_episode', required=False, action='store', dest='episode',
                        help='Load data from idicated episode')
    parser.add_argument('-x', '--experiment', required=True, action='store', dest='experiment', type=int,
                        help='Setup experiment')
    parser.add_argument('-g', '--gpu', required=False, action='store_true', dest='gpu',
                        help='Use GPU (with fixed memory limit to prevent crashes).')
    parser.add_argument('-p', '--port', required=False, action='store', default=PORT, dest='port',
                        help='Listen at port')

    args = parser.parse_args()
    args.port = int(args.port)

    if args.episode:
        args.episode = int(args.episode)

    return args


def set_fixed_memory():
    """
    Fixed memory limit to prevent crash.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])


def wait_for_connection(sock, port):
    """
    Open port for incoming game status.
    """
    # Allow reuse port. Useful for testing, when server is killed many times
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((HOST, port))
    sock.listen()
    print("Starting server at %s:%d" % (HOST, port))

    return sock


def load_snapshot_metadata(episode, agent):
    """
    Load snapshot metadata.
    """
    if episode:
        with open(DATA_SNAPSHOT % episode, "rb") as f:
            total_round, epsilon, replay_memory, _ = pickle.load(f)
            agent.load_replay_memory(replay_memory)
        start_episode = episode + 1
    else:
        epsilon = 1
        start_episode = 0
        total_round = 0

    return start_episode, total_round, epsilon


def save_snapshot(agent, epsilon, episode, episode_reward, total_round, moves):
    """Save snapshot."""
    agent.get_tf_model().save(MODEL_SNAPSHOT % episode)

    with open(DATA_SNAPSHOT % episode, "wb") as f:
        pickle.dump((total_round, epsilon, agent.replay_memory, episode_reward), f)

    with open(STATS_FILE, "a") as f:
        f.write("Episode: %d, round: %d, epsilon: %0.2f, moves: %d, reward: %0.2f\n"
            % (episode, total_round, epsilon, moves, episode_reward))


def start_learning(sock, start_episode, total_round, epsilon, play_one_game, agent):
    """
    Learn through episodes.
    """
    env = Environment(sock)

    for episode in range(start_episode, EPISODES + 1):
        total_round, episode_reward, epsilon = play_one_game(total_round, epsilon, env, agent)

        if episode > 0 and episode % SNAPSHOT_MODULO == 0:
            save_snapshot(agent, epsilon, episode, episode_reward, total_round, len(env.handling_time))

        print("Episode %d, round: %d, epsilon %0.3f, reward %0.2f, moves %d, avg handling time: %0.4f, game time: %0.4f"
            % (episode,
                total_round,
                epsilon,
                episode_reward,
                len(env.handling_time),
                sum(env.handling_time) / len(env.handling_time),
                time.time() - env.game_tic))

    env.close()
