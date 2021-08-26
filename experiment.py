#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""

import socket
import argparse
from robot.models.flatnn_model import FlatNnModel
from robot.models.cnn_model import CnnModel
from robot.models.sp_model import SpModel
from robot.agents.agent import Agent
from robot.agents.caching_agent import CachingAgent
from robot.learners import simple
from robot.learners import simple_with_solver
from robot.learners import inter_scoring
from robot.learners import inter_scoring_cache
from robot.learners import simple_episode_espsilon
from robot.learners import sp
from robot import config
from robot import utils
from robot import training


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
                'Robot is waiting for connection from Agent at ' + config.HOST + ':' + str(config.PORT) + '\n',
        usage='Please try to use -h, --help for more informations',
        epilog=' \n',
        formatter_class=CustomFormatter)

    parser.add_argument('-e', '--load_episode', required=False, action='store', dest='episode',
                        help='Load data from idicated episode.')
    parser.add_argument('-x', '--experiment', required=True, action='store', dest='experiment', type=int,
                        help='Setup experiment.')
    parser.add_argument('-g', '--gpu', required=False, action='store_true', dest='gpu',
                        help='Use GPU (with fixed memory limit to prevent crashes).')
    parser.add_argument('-p', '--port', required=False, action='store', default=config.PORT, dest='port',
                        help='Listen at port.')
    parser.add_argument('-d', '--disable-learning', required=False, action='store_true', dest='disable_learning',
                        help='Disable learning. All actions from model without update.')

    args = parser.parse_args()
    args.port = int(args.port)
    args.enable_learning = not args.disable_learning

    if args.episode:
        args.episode = int(args.episode)

    return args


if __name__ == '__main__':
    args = parse_args()

    if args.gpu:
        utils.set_fixed_memory()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        utils.wait_for_connection(sock, args.port)

        if args.experiment == 1:
            utils.log_in_stats("Experiment: %d. Flat NN." % args.experiment)

            model = FlatNnModel(args.episode)
            agent = Agent(model)
            play_one_game = simple.play_one_game
        elif args.experiment == 2:
            utils.log_in_stats("Experiment: %d. Convolutional NN." % args.experiment)

            model = CnnModel(args.episode)
            agent = Agent(model)
            play_one_game = simple.play_one_game
        elif args.experiment == 3:
            assert sp.UPDATE_MODEL_ROUND % config.SNAPSHOT_MODULO == 0, \
                "Caching and training model can't differ when snapshot is saved"
            utils.log_in_stats("Experiment: %d. Stevens and Pradhan robot" % args.experiment)

            training_model = SpModel(args.episode)
            caching_model = SpModel(args.episode)
            agent = CachingAgent(training_model, caching_model)
            play_one_game = sp.play_one_game
        elif args.experiment == 4:
            utils.log_in_stats("Experiment: %d. Flat NN and scoring based on mistakes." % args.experiment)

            model = FlatNnModel(args.episode)
            agent = Agent(model)
            play_one_game = inter_scoring.play_one_game
        elif args.experiment == 5:
            assert inter_scoring_cache.UPDATE_MODEL_ROUND % config.SNAPSHOT_MODULO == 0, \
                "Caching and training model can't differ when snapshot is saved"
            utils.log_in_stats("Experiment: %d. Flat NN with caching and with scoring based on mistakes." % args.experiment)

            training_model = FlatNnModel(args.episode)
            caching_model = FlatNnModel(args.episode)
            agent = CachingAgent(training_model, caching_model)
            play_one_game = inter_scoring_cache.play_one_game
        elif args.experiment == 6:
            utils.log_in_stats("Experiment: %d. Flat NN with solver" % args.experiment)

            model = FlatNnModel(args.episode)
            agent = Agent(model)
            play_one_game = simple_with_solver.play_one_game
        elif args.experiment == 7:
            utils.log_in_stats("Experiment: %d. Flat NN with epsilon calculated after episode" % args.experiment)

            model = FlatNnModel(args.episode)
            agent = Agent(model)
            play_one_game = simple_episode_espsilon.play_one_game
        else:
            raise Exception("Experiment %d is missing. Please check documentation." % args.experiment)

        start_episode, total_round, epsilon = utils.load_snapshot_metadata(args.episode, agent)
        training.start(sock, start_episode, total_round, epsilon, play_one_game, agent, args.enable_learning)
