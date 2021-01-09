#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""


"""
Reinforcement learning - Deep Q-Network/Learning.

Credits:
- "Playing Atari with Deep Reinforcement Learning" by
    Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves,
    Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller; DeepMind Technologies
    https://arxiv.org/pdf/1312.5602.pdf
- "Human-level control through deep reinforcement learning" by
    Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness,
    Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland,
    Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou,
    Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg & Demis Hassabis
    https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
- "Training Deep Q Learning and Deep Q Networks (DQN) Intro and Agent -
    Reinforcement Learning w/ Python Tutorial" by Harrison Kinsley
    https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/
- https://www.quora.com/Artificial-Intelligence-What-is-an-intuitive-explanation-of-how-deep-Q-networks-DQN-work
- "How to use Q Learning in Video Games Easily" by Siraj Raval
    https://www.youtube.com/watch?v=A5eihauRQvo
- Q-learning
    https://en.wikipedia.org/wiki/Q-learning#Algorithm
- Bellman equation
    https://en.wikipedia.org/wiki/Bellman_equation
"""

import tensorflow as tf
import numpy as np
import socket
import time
import argparse
import pickle
from flatnn_model import FlatNnModel
from envirement import Environment
from heuristic_solver import HeuristicSolver
from agent import Agent
import config


def main():
    args = parse_args()

    # Fixed memory limit to prevent crash
    if args.gpu:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        agent, epsilon, start_episode = create_agent(args.episode)

        # Allow reuse port. Useful for testing, when server is killed many times
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((config.HOST, args.port))
        sock.listen()
        print("Starting server at %s:%d" % (config.HOST, args.port))
        env = Environment(sock)

        start_learning(env, epsilon, agent, start_episode)

        env.close()


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def parse_args():
    """Parse command line arguments."""
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
                        help='Load data from idicated episode')
    parser.add_argument('-g', '--gpu', required=False, action='store_true', dest='gpu',
                        help='Use GPU (with fixed memory limit to prevent crashes).')
    parser.add_argument('-p', '--port', required=False, action='store', default=config.PORT, dest='port',
                        help='Listen at port')

    args = parser.parse_args()
    args.port = int(args.port)

    if args.episode:
        args.episode = int(args.episode)

    return args


def create_agent(episode):
    """Create agent from existing snapshot, or create new one."""
    model = FlatNnModel(episode)
    agent = Agent(model)

    if episode:
        with open(config.DATA_SNAPSHOT % episode, "rb") as f:
            epsilon, agent.replay_memory, _ = pickle.load(f)
        start_episode = episode + 1
    else:
        epsilon = 1
        start_episode = 0

    return agent, epsilon, start_episode


def start_learning(env, epsilon, agent, start_episode):
    """Learn through episodes."""
    for episode in range(start_episode, config.EPISODES + 1):
        episode_reward, epsilon = play_one_game(epsilon, env, agent)

        if episode > 0 and episode % config.SNAPSHOT_MODULO == 0:
            save_snapshot(agent, epsilon, episode, episode_reward, len(env.handling_time))

        print("Episode %d, epsilon %0.3f, reward %0.2f, moves %d, avg handling time: %0.4f, game time: %0.4f"
            % (episode,
                epsilon,
                episode_reward,
                len(env.handling_time),
                sum(env.handling_time) / len(env.handling_time),
                time.time() - env.game_tic))


def play_one_game(epsilon, env, agent):
    """Play one game."""
    episode_reward = 0
    solver = HeuristicSolver()

    # Reset environment and get initial state
    _, _, current_piece, raw_current_state, current_state = env.reset()
    current_state = agent.reshape_input(current_state)

    # Reset flag and start iterating until episode ends
    last_round = False

    while not last_round:
        # Explore other actions with probability epsilon
        if np.random.random() <= epsilon:
            action = solver.action(current_piece, raw_current_state)
        else:
            q_values = agent.q_values_for_state(current_state)
            # Choose best action
            action = np.argmax(q_values)

        last_round, reward, next_piece, raw_next_state, next_state = env.step(action)
        next_state = agent.reshape_input(next_state)

        # Transform new continuous state to new discrete state and count reward
        episode_reward += reward

        # Every step update replay memory and train NN model
        transition = Transition(current_state, action, reward, next_state, last_round)
        agent.update_replay_memory(transition)
        agent.train(last_round)

        current_piece = next_piece
        current_state = next_state
        raw_current_state = raw_next_state

        epsilon = adjust_epsilon(epsilon)

    return episode_reward, epsilon


def adjust_epsilon(epsilon):
    """Decay epsilon."""
    if epsilon > config.MIN_EPSILON:
        epsilon = epsilon * config.EPSILON_DECAY

    return epsilon


class Transition:
    def __init__(self, current_state, action, reward, next_state, last_round):
        self.current_state = current_state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.last_round = last_round


def save_snapshot(agent, epsilon, episode, episode_reward, moves):
    """Save snapshot."""
    agent.get_tf_model().save(config.MODEL_SNAPSHOT % episode)

    with open(config.DATA_SNAPSHOT % episode, "wb") as f:
        pickle.dump((epsilon, agent.replay_memory, episode_reward), f)

    with open(config.STATS_FILE, "a") as f:
        f.write("Episode: %d, epsilon: %0.2f, moves: %d, reward: %0.2f\n" % (episode, epsilon, moves, episode_reward))


if __name__ == '__main__':
    main()
