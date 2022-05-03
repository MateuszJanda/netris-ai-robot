#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""

import socket
import time
from robot.environments.proxy_environment import ProxyEnvironment
from robot.environments.local_environment import LocalEnvironment
from robot import config
from robot import utils


class Training:
    def __init__(self, args, play_one_game_func, agent) -> None:
        self.args = args
        self.play_one_game_func = play_one_game_func
        self.agent = agent

    def start(self):
        """
        Create environment and start learning.
        """
        if self.args.local_env:
            print("[!] Run LocalEnvironment")
            env = LocalEnvironment()
            self._start_environment(env)
        elif self.args.proxy_env_port:
            print("[!] Run ProxyEnvironment")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                self._wait_for_connection(sock, self.args.proxy_env_port)
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
        print("[!] Starting server for proxy agent at %s:%d" % (config.HOST, port))

        return sock

    def _start_environment(self, env):
        """
        Start learning through episodes.
        """
        # Load snapshot if available
        start_episode, total_rounds, epsilon = utils.load_snapshot_metadata(self.args.episode, self.agent)

        for episode in range(start_episode, config.EPISODES + 1):
            total_rounds, episode_reward, episode_lines, epsilon = self.play_one_game_func(total_rounds,
                epsilon, env, self.agent, self.args.enable_learning)

            # Save snapshot if right episode
            utils.save_snapshot(episode, self.agent, total_rounds, epsilon, episode_reward, episode_lines)
            # Print board if right episode
            utils.print_board(episode, env)

            # Calculate stats
            game_time = time.time() - env.game_tic
            moves = len(env.handle_times)
            one_step_time = sum(env.handle_times) / moves
            utils.save_stats(episode, total_rounds, epsilon, episode_reward, episode_lines, moves, one_step_time, game_time)

        env.close()
