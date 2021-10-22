#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""

import argparse
from robot.models.flatnn_model import FlatNnModel
from robot.models.cnn_model import CnnModel
from robot.models.sp_model import SpModel
from robot.agents.agent import Agent
from robot.agents.caching_agent import CachingAgent
from robot.strategy import simple_strategy
from robot.strategy import simple_with_solver_strategy
from robot.strategy import inter_scoring_strategy
from robot.strategy import inter_scoring_cache_strategy
from robot.strategy import simple_episode_espsilon_strategy
from robot.strategy import sp_strategy
from robot.training import Training
from robot import config
from robot import utils


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
                'If -p/--proxy-agent-port is passed robot is waiting for connection\n' +
                'from proxy agent at ' + config.HOST + ':' + '<PORT>\n',
        usage='Please try to use -h, --help for more informations',
        epilog=' \n',
        formatter_class=CustomFormatter)

    parser.add_argument('-e', '--load-episode', required=False, action='store', dest='episode',
                        help='Load data from idicated episode.')
    parser.add_argument('-x', '--experiment', required=True, action='store', dest='experiment', type=int,
                        help='Setup experiment.')
    parser.add_argument('-g', '--use-gpu', required=False, action='store_true', dest='use_gpu',
                        help='Use GPU (with fixed memory limit to prevent crashes).')
    parser.add_argument('-p', '--proxy-agent-port', required=False, action='store', dest='proxy_agent_port', type=int,
                        help='Run proxy envirement, and listen at port from proxy agent. If this paramet not passed ' \
                            'local envirement is used.')
    parser.add_argument('-d', '--disable-learning', required=False, action='store_true', dest='disable_learning',
                        help='Disable learning. All actions are predicted from model.')

    args = parser.parse_args()
    args.enable_learning = not args.disable_learning

    if args.episode:
        args.episode = int(args.episode)

    return args


if __name__ == '__main__':
    args = parse_args()

    if args.use_gpu:
        utils.set_fixed_memory()

    if args.experiment == 1:
        utils.log_in_stats("Experiment: %d. Flat NN." % args.experiment)

        model = FlatNnModel(args.episode)
        agent = Agent(model)
        play_one_game_func = simple_strategy.play_one_game
    elif args.experiment == 2:
        utils.log_in_stats("Experiment: %d. Convolutional NN." % args.experiment)

        model = CnnModel(args.episode)
        agent = Agent(model)
        play_one_game_func = simple_strategy.play_one_game
    elif args.experiment == 3:
        assert sp_strategy.UPDATE_MODEL_ROUND % config.SNAPSHOT_MODULO == 0, \
            "Caching and training model can't differ when snapshot is saved"
        utils.log_in_stats("Experiment: %d. Stevens and Pradhan robot" % args.experiment)

        training_model = SpModel(args.episode)
        caching_model = SpModel(args.episode)
        agent = CachingAgent(training_model, caching_model)
        play_one_game_func = sp_strategy.play_one_game
    elif args.experiment == 4:
        utils.log_in_stats("Experiment: %d. Flat NN and scoring based on mistakes." % args.experiment)

        model = FlatNnModel(args.episode)
        agent = Agent(model)
        play_one_game_func = inter_scoring_strategy.play_one_game
    elif args.experiment == 5:
        assert inter_scoring_cache_strategy.UPDATE_MODEL_ROUND % config.SNAPSHOT_MODULO == 0, \
            "Caching and training model can't differ when snapshot is saved"
        utils.log_in_stats("Experiment: %d. Flat NN with caching and with scoring based on mistakes." % args.experiment)

        training_model = FlatNnModel(args.episode)
        caching_model = FlatNnModel(args.episode)
        agent = CachingAgent(training_model, caching_model)
        play_one_game_func = inter_scoring_cache_strategy.play_one_game
    elif args.experiment == 6:
        utils.log_in_stats("Experiment: %d. Flat NN with solver" % args.experiment)

        model = FlatNnModel(args.episode)
        agent = Agent(model)
        play_one_game_func = simple_with_solver_strategy.play_one_game
    elif args.experiment == 7:
        utils.log_in_stats("Experiment: %d. Flat NN with epsilon calculated after episode" % args.experiment)

        model = FlatNnModel(args.episode)
        agent = Agent(model)
        play_one_game_func = simple_episode_espsilon_strategy.play_one_game
    else:
        raise Exception("Experiment %d is missing. Please check documentation." % args.experiment)

    training = Training(args, play_one_game_func, agent)
    training.start()
