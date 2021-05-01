#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""


"""
Reinforcement learning - Deep Q-Network/Learning. Simple flat NN with NetrisSolver.
"""

import numpy as np
from dqn.netris_solver import NetrisSolver
from dqn import config


EPSILON_DECAY = 0.99995     # Decay epsilon. Smarter NN is, then less random action should be taken
MIN_EPSILON = 0.02          # Epsilon shouldn't less than this. We always want to check something new
RAND_TRESHOLD = 0.005


def play_one_game(total_rounds, epsilon, env, agent):
    """
    Play one game.
    """
    episode_reward = 0
    episode_lines = 0
    solver = NetrisSolver()

    # Reset environment and get initial state
    _, _, current_piece, raw_current_state, current_state = env.reset()

    # Reset flag and start iterating until episode ends
    last_round = False

    while not last_round:
        # Explore other actions with probability epsilon
        r = np.random.random()
        if r < epsilon:
            if r < RAND_TRESHOLD:
                action = np.random.randint(0, config.ACTION_SPACE_SIZE)
            else:
                action = solver.action(current_piece, raw_current_state)
        else:
            q_values = agent.q_values_for_state(current_state)
            # Choose best action
            action = np.argmax(q_values)

            if np.isnan(q_values[action]) or np.isinf(q_values[action]):
                print("Error: Q value = ", q_values[action])
                exit()

        last_round, lines, next_piece, raw_next_state, next_state = env.step(action)
        episode_lines += lines

        # Transform new continuous state to new discrete state and count reward
        reward = adjust_reward(lines)
        episode_reward += reward

        # Every step update replay memory and train NN model
        transition = config.Transition(current_state, action, reward, next_state, last_round)
        agent.update_replay_memory(transition)
        agent.train(last_round)

        current_piece = next_piece
        current_state = next_state
        raw_current_state = raw_next_state

        epsilon = adjust_epsilon(epsilon)
        total_rounds += 1

    return total_rounds, episode_reward, episode_lines, epsilon


def adjust_reward(lines):
    """
    Adjust reward - lines_cleared**2
    """
    return lines**2


def adjust_epsilon(epsilon):
    """
    Decay epsilon.
    """
    if epsilon > MIN_EPSILON:
        epsilon = epsilon * EPSILON_DECAY

    return epsilon
