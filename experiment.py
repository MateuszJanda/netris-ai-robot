#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""

import socket
from robot.flatnn_model import FlatNnModel
from robot.cnn_model import CnnModel
from robot.sp_model import SpModel
from robot.agent import Agent
from robot.caching_agent import CachingAgent
from robot import simple_robot
from robot import simple_with_solver_robot
from robot import inter_scoring_robot
from robot import inter_scoring_cache_robot
from robot import simple_episode_espsilon
from robot import sp_robot
from robot import config


if __name__ == '__main__':
    args = config.parse_args()

    if args.gpu:
        config.set_fixed_memory()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        config.wait_for_connection(sock, args.port)

        if args.experiment == 1:
            config.log_in_stats("Experiment: %d. Flat NN." % args.experiment)

            model = FlatNnModel(args.episode)
            agent = Agent(model)
            play_one_game = simple_robot.play_one_game
        elif args.experiment == 2:
            config.log_in_stats("Experiment: %d. Convolutional NN." % args.experiment)

            model = CnnModel(args.episode)
            agent = Agent(model)
            play_one_game = simple_robot.play_one_game
        elif args.experiment == 3:
            assert sp_robot.UPDATE_MODEL_ROUND % config.SNAPSHOT_MODULO == 0, \
                "Caching and training model can't differ when snapshot is saved"
            config.log_in_stats("Experiment: %d. Stevens and Pradhan robot" % args.experiment)

            training_model = SpModel(args.episode)
            caching_model = SpModel(args.episode)
            agent = CachingAgent(training_model, caching_model)
            play_one_game = sp_robot.play_one_game
        elif args.experiment == 4:
            config.log_in_stats("Experiment: %d. Flat NN and scoring based on mistakes." % args.experiment)

            model = FlatNnModel(args.episode)
            agent = Agent(model)
            play_one_game = inter_scoring_robot.play_one_game
        elif args.experiment == 5:
            assert inter_scoring_cache_robot.UPDATE_MODEL_ROUND % config.SNAPSHOT_MODULO == 0, \
                "Caching and training model can't differ when snapshot is saved"
            config.log_in_stats("Experiment: %d. Flat NN with caching and with scoring based on mistakes." % args.experiment)

            training_model = FlatNnModel(args.episode)
            caching_model = FlatNnModel(args.episode)
            agent = CachingAgent(training_model, caching_model)
            play_one_game = inter_scoring_cache_robot.play_one_game
        elif args.experiment == 6:
            config.log_in_stats("Experiment: %d. Flat NN with solver" % args.experiment)

            model = FlatNnModel(args.episode)
            agent = Agent(model)
            play_one_game = simple_with_solver_robot.play_one_game
        elif args.experiment == 7:
            config.log_in_stats("Experiment: %d. Flat NN with epsilon calculated after episode" % args.experiment)

            model = FlatNnModel(args.episode)
            agent = Agent(model)
            play_one_game = simple_episode_espsilon.play_one_game
        else:
            raise Exception("Experiment %d is missing, check docs" % args.experiment)

        start_episode, total_round, epsilon = config.load_snapshot_metadata(args.episode, agent)
        config.start_learning(sock, start_episode, total_round, epsilon, play_one_game, agent, args.enable_learning)
