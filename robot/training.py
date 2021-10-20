#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""

import socket
import time
from robot.environments.proxy_environment import ProxyEnvironment
from robot import config
from robot import utils


def wait_for_connection(sock, port):
    """
    Open port for incoming game status.
    """
    # Allow port reuse. Useful for testing, when server is killed many times
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((config.HOST, port))
    sock.listen()
    print("Starting server at %s:%d" % (config.HOST, port))

    return sock


def start(args, play_one_game, agent):
    """
    Start learning through episodes.
    """

    start_episode, total_rounds, epsilon = utils.load_snapshot_metadata(args.episode, agent)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        wait_for_connection(sock, args.port)

        env = ProxyEnvironment(sock)

        for episode in range(start_episode, config.EPISODES + 1):
            total_rounds, episode_reward, episode_lines, epsilon = play_one_game(total_rounds,
                epsilon, env, agent, args.enable_learning)

            utils.save_snapshot(agent, episode, total_rounds, epsilon, episode_reward, episode_lines)

            # Calculate stats
            game_time = time.time() - env.game_tic
            moves = len(env.handle_times)
            one_step_time = sum(env.handle_times) / moves
            utils.save_stats(episode, total_rounds, epsilon, episode_reward, episode_lines, moves, one_step_time, game_time)

        env.close()
