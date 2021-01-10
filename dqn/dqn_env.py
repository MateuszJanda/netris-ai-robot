#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""


"""
Reinforcement learning - Deep Q-Network/Learning.
"""

import numpy as np
import socket
from flatnn_model import FlatNnModel
from heuristic_solver import HeuristicSolver
import config


def play_one_game(epsilon, env, agent):
    """
    Play one game.
    """
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
        transition = config.Transition(current_state, action, reward, next_state, last_round)
        agent.update_replay_memory(transition)
        agent.train(last_round)

        current_piece = next_piece
        current_state = next_state
        raw_current_state = raw_next_state

        epsilon = adjust_epsilon(epsilon)

    return episode_reward, epsilon


def adjust_epsilon(epsilon):
    """
    Decay epsilon.
    """
    if epsilon > config.MIN_EPSILON:
        epsilon = epsilon * config.EPSILON_DECAY

    return epsilon


if __name__ == '__main__':
    args = config.parse_args()

    if args.gpu:
        config.set_fixed_memory()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        config.wait_for_connection(sock, args.port)
        config.start_learning(sock, args.episode, play_one_game, model=FlatNnModel)
