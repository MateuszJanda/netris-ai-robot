#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""

import socket
from dqn.flatnn_model import FlatNnModel
from dqn import simple_dqn
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
            config.start_learning(sock, args.episode, simple_dqn.play_one_game, model)
        else:
            print("Load default experiment")
            model = FlatNnModel(args.episode)
            config.start_learning(sock, args.episode, simple_dqn.play_one_game, model)
