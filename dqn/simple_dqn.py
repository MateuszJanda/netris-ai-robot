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
from dqn import config


# Exploration settings - try/explore random action with probability epsilon
EPSILON_DECAY = 0.99995     # Decay epsilon. Smarter NN is, then less random action should be taken
MIN_EPSILON = 0.02          # Epsilon shouldn't less than this. We always want to check something new


def play_one_game(total_rounds, epsilon, env, agent, enable_learning):
    """
    Play one game.
    """
    episode_reward = 0
    episode_lines = 0

    # Reset environment and get initial state
    _, _, _, _, current_state = env.reset()

    # Reset flag and start iterating until episode ends
    last_round = False

    while not last_round:
        # Explore other actions with probability epsilon
        if enable_learning and np.random.random() <= epsilon:
            action = np.random.randint(0, config.ACTION_SPACE_SIZE)
        else:
            q_values = agent.q_values_for_state(current_state)
            # Choose best action
            action = np.argmax(q_values)

        last_round, lines, _, _, next_state = env.step(action)
        episode_lines += lines

        # Transform new continuous state to new discrete state and count reward
        reward = adjust_reward(lines)
        episode_reward += reward

        # Every step update replay memory and train NN model
        if enable_learning:
            transition = config.Transition(current_state, action, reward, next_state, last_round)
            agent.update_replay_memory(transition)
            agent.train(last_round)

        current_state = next_state

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
