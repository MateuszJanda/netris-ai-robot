#!/usr/bin/env python3

# Author: Mateusz Janda <mateusz janda at gmail com>
# Site: github.com/MateuszJanda/netris-ai-robot
# Ad maiorem Dei gloriam

"""
Setup environment and start training.
"""

import socket
from robot.environments.proxy_environment import ProxyEnvironment
from robot.environments.local_environment import LocalEnvironment
from robot import config
from robot import utils


class Training:
    """
    Class is responsible for environment setup and starting training.
    """

    def __init__(self, args, strategy, agent):
        self._args = args
        self._strategy = strategy
        self._agent = agent

    def start(self):
        """
        Create environment and start learning.
        """
        if self._args.local_env:
            print("[!] Run LocalEnvironment")
            env = LocalEnvironment()
            self._start_environment(env)
        elif self._args.proxy_env_port:
            print("[!] Run ProxyEnvironment")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                self._wait_for_connection(sock, self._args.proxy_env_port)
                env = ProxyEnvironment(sock)
                self._start_environment(env)

    def _wait_for_connection(self, sock, port):
        """
        Open port for incoming game status.
        """
        # Allow port reuse. Useful for testing, when server is killed many times
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((config.HOST, port))
        sock.listen()
        print(f"[!] Starting server for proxy agent at {config.HOST}:{port}")

        return sock

    def _start_environment(self, env):
        """
        Start learning through episodes.
        """
        # Load snapshot if available
        start_episode, total_steps, epsilon = utils.load_snapshot_metadata(self._args.episode, self._agent)

        for episode in range(start_episode, config.EPISODES + 1):
            total_steps, episode_reward, episode_lines, epsilon = self._strategy.play(total_steps,
                epsilon, env, self._agent, self._args.enable_learning)

            # Save snapshot if right episode
            if episode > 0 and episode % config.SNAPSHOT_MODULO == 0:
                utils.save_snapshot(episode, self._agent, total_steps, epsilon, episode_reward, episode_lines)

            # Print board if right episode
            if episode > 0 and episode % config.PRINT_BOARD_MODULO == 0:
                utils.print_board(episode, env)

            utils.save_stats(episode, total_steps, epsilon, episode_reward, episode_lines, env)

            # If model is good enough then finish
            if epsilon < 0.1 and episode_lines >= config.MAX_LINES_IN_EPISODE:
                utils.save_snapshot(episode, self._agent, total_steps, epsilon, episode_reward, episode_lines)
                utils.log_in_stats("[!] Model is good enough. Finish.")
                break

        env.close()
