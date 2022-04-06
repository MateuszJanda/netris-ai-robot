#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""

import tensorflow as tf
import socket
import time
import pickle
from robot import config


class Transition:
    def __init__(self, current_state, action, reward, next_state, last_round):
        self.current_state = current_state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.last_round = last_round


def set_fixed_memory():
    """
    Fixed memory limit to prevent crash.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])


def load_snapshot_metadata(episode, agent):
    """
    Load snapshot metadata.
    """
    if episode:
        with open(config.DATA_SNAPSHOT % episode, "rb") as f:
            total_rounds, epsilon, replay_memory, _, _ = pickle.load(f)
            agent.load_replay_memory(replay_memory)
        start_episode = episode + 1
    else:
        epsilon = 1
        start_episode = 0
        total_rounds = 0

    return start_episode, total_rounds, epsilon


def save_snapshot(agent, episode, total_rounds, epsilon, episode_reward, episode_lines):
    """Save snapshot."""
    if episode > 0 and episode % config.SNAPSHOT_MODULO == 0:
        agent.get_tf_model().save(config.MODEL_SNAPSHOT % episode)

        with open(config.DATA_SNAPSHOT % episode, "wb") as f:
            pickle.dump((total_rounds, epsilon, agent.replay_memory, episode_reward, episode_lines), f)


def save_stats(episode, total_rounds, epsilon, episode_reward, episode_lines, moves, one_step_time, game_time):
    """
    Log and save statistics.
    """
    with open(config.STATS_FILE, "a") as f:
        f.write("Episode: %d, rounds: %d, epsilon: %0.2f, reward: %0.2f, lines: %d, moves: %d\n"
            % (episode, total_rounds, epsilon, episode_reward, episode_lines, moves))

    print("[+] Episode: %d, rounds: %d, epsilon: %0.3f, reward: %0.2f, lines: %d, moves: %d, one step time: %0.4f, game time: %0.4f"
        % (episode,
            total_rounds,
            epsilon,
            episode_reward,
            episode_lines,
            moves,
            one_step_time,
            game_time))


def log_in_stats(text):
    """
    Log in stats file.
    """
    with open(config.STATS_FILE, "a") as f:
        f.write(text + "\n")

    print('[+]' + text)
