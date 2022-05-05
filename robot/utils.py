#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""

import tensorflow as tf
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
    gpus = tf.config.experimental.list_physical_devices("GPU")

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4*1024)])


def load_snapshot_metadata(episode, agent):
    """
    Load snapshot metadata.
    """
    if episode:
        with open(config.DATA_SNAPSHOT % episode, "rb") as f:
            total_steps, epsilon, replay_memory, _, _ = pickle.load(f)
            agent.load_replay_memory(replay_memory)
        start_episode = episode + 1
    else:
        epsilon = 1
        start_episode = 0
        total_steps = 0

    return start_episode, total_steps, epsilon


def save_snapshot(episode, agent, total_steps, epsilon, episode_reward, episode_lines):
    """Save snapshot."""
    if episode > 0 and episode % config.SNAPSHOT_MODULO == 0:
        agent.get_tf_model().save(config.MODEL_SNAPSHOT % episode)

        with open(config.DATA_SNAPSHOT % episode, "wb") as f:
            pickle.dump((total_steps, epsilon, agent.replay_memory, episode_reward, episode_lines), f)


def save_stats(episode, total_steps, epsilon, episode_reward, episode_lines, env):
    """
    Log and save statistics.
    """
    with open(config.STATS_FILE, "a") as f:
        f.write(f"Episode: {episode}, total steps: {total_steps}, epsilon: {epsilon: 0.2f}, " \
                f"reward: {episode_reward: 0.2f}, lines: {episode_lines}, steps: {env.num_of_steps()}\n")

    print(f"[+] Episode: {episode}, total steps: {total_steps}, epsilon: {epsilon:0.3f}, " \
          f"reward: {episode_reward:0.2f}, lines: {episode_lines}, steps: {env.num_of_steps()}, " \
          f"step duration: {env.step_duration():0.4f}, game duration: {env.game_duration():0.4f}")


def print_board(episode, env):
    """Print board from raw_board (without piece)."""
    if episode == 0 or episode % config.PRINT_BOARD_MODULO != 0:
        return

    print("[i] Board at episode:", episode)
    look = ""
    board = env.raw_board()
    for line in board:
        blocks_line = "".join(["1" if b else " " for b in line])
        look += "|" + blocks_line + "|\n"

    print(look)


def log_in_stats(text):
    """
    Log in stats file.
    """
    with open(config.STATS_FILE, "a") as f:
        f.write(text + "\n")

    print("[+]" + text)
