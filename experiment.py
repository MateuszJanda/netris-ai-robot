#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""

import socket
from dqn.flatnn_model import FlatNnModel
from dqn.cnn_model import CnnModel
from dqn.sp_model import SpModel
from dqn.agent import Agent
from dqn.caching_agent import CachingAgent
from dqn import simple_dqn
from dqn import sp_dqn
from dqn import config


if __name__ == '__main__':
    args = config.parse_args()

    if args.gpu:
        config.set_fixed_memory()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        config.wait_for_connection(sock, args.port)

        if args.experiment == 1:
            print("Experiment: %d. Flat NN." % args.experiment)

            model = FlatNnModel(args.episode)
            agent = Agent(model)
            start_episode, total_round, epsilon = config.load_snapshot_metadata(args.episode, agent)

            config.start_learning(sock, start_episode, total_round, epsilon, simple_dqn.play_one_game, agent)
        elif args.experiment == 2:
            print("Experiment: %d. Convolutional NN." % args.experiment)

            model = CnnModel(args.episode)
            agent = Agent(model)
            start_episode, total_round, epsilon = config.load_snapshot_metadata(args.episode, agent)

            config.start_learning(sock, start_episode, total_round, epsilon, simple_dqn.play_one_game, agent)
        elif args.experiment == 3:
            print("Experiment: %d. Stevens and Pradhan DQN" % args.experiment)

            model = SpModel(args.episode)
            agent = CachingAgent(model)
            start_episode, total_round, epsilon = config.load_snapshot_metadata(args.episode, agent)

            config.start_learning(sock, start_episode, total_round, epsilon, sp_dqn.play_one_game, agent)
