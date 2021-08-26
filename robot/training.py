#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""

from robot.environment.environment import Environment
from robot import config
from robot import utils
import time


def start_learning(sock, start_episode, total_rounds, epsilon, play_one_game, agent, enable_learning):
    """
    Learn through episodes.
    """
    env = Environment(sock)

    for episode in range(start_episode, config.EPISODES + 1):
        total_rounds, episode_reward, episode_lines, epsilon = play_one_game(total_rounds,
            epsilon, env, agent, enable_learning)

        utils.save_snapshot(agent, episode, total_rounds, epsilon, episode_reward, episode_lines)

        game_time = time.time() - env.game_tic
        moves = len(env.handling_time)
        avg_handling_time = sum(env.handling_time) / moves
        utils.save_stats(episode, total_rounds, epsilon, episode_reward, episode_lines, moves, avg_handling_time, game_time)

    env.close()